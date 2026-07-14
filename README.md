# MADA Competition Evaluator

Python 后端评分服务，用于接收 q1/q2 目标检测、q3 苹果糖酸度或 q4 图像计数预测结果，和本地 ground truth 比对并异步写回数据库。

## 接口

启动：

```bash
uvicorn evaluator_service.main:app --host 0.0.0.0 --port 8000
```

请求：

```http
POST /submissions
Content-Type: application/json
```

```json
{
  "submission_id": "uuid-of-competition-prediction-submission",
  "topic_id": 1,
  "file_name": "prediction.zip",
  "file_size": 12345678,
  "download_url": "https://example.com/presigned-download-url"
}
```

通过同步校验后返回：

```json
{
  "accepted": true,
  "job_id": "optional-evaluator-job-id",
  "message": "格式校验通过，已进入评分队列"
}
```

不通过返回：

```json
{
  "accepted": false,
  "error": "提交缺少 1 个 label 文件: example.txt"
}
```

## 提交格式

- `topic_id=1` 使用 `q1/labels` 作为真值。
- `topic_id=2` 使用 `q2/labels/val` 作为真值。
- q1/q2 必须上传 zip；zip 内可以直接放 `.txt`，也可以放 `labels/` 或 `<任意目录>/labels/`。
- `.txt` 文件名集合必须和对应赛题真值完全一致。
- 真值格式为 YOLO: `class x_center y_center width height`。
- 预测格式支持 `class x_center y_center width height confidence`；如果只有 5 列，服务会按 `confidence=1.0` 处理。
- `topic_id=3` 使用 `q3/val_result.xlsx` 作为真值，直接上传一个 `.xlsx`，不使用 zip。
- q3 工作簿必须且只能包含一个工作表，表头严格为 `序号`、`糖度`、`酸度`。
- q3 按 `序号` 关联预测和真值；序号集合必须完全一致，且糖度、酸度必须是非负有限数值，不允许公式。
- `topic_id=4` 使用 `q4/val.txt` 作为真值，直接上传一个 UTF-8 `.txt`，不使用 zip。
- q4 每个非空行严格为 `图片文件名 预测数量` 两列；按图片文件名关联，提交集合必须与真值完全一致。
- q4 预测数量允许非负有限小数，图片行顺序不影响评分。

## 指标

- q1: 计算 `mAP50` 和 `mAP50_95`，`score = 0.5 * mAP50 + 0.5 * mAP50_95`。
- q2: 计算 `mAP50_95`，`score = mAP50_95`。
- q3: 分别计算糖度和酸度的归一化绝对误差，再取等权平均：

```text
sugar_error = sum(abs(predicted_sugar - true_sugar)) / sum(true_sugar)
acid_error = sum(abs(predicted_acid - true_acid)) / sum(true_acid)
error = score = 0.5 * sugar_error + 0.5 * acid_error
```

- q3 的验证集和测试集各自使用当前 ground truth 的真值总和作为分母，不进行 40%/60% 合并。
- q3 的 `score` 越小越好，完美预测为 `0`；排行榜必须对 q3 按 `score` 升序排列。
- q4 使用所有图片预测数量的均方误差：

```text
mse = score = sum((predicted_count - true_count) ** 2) / image_count
```

- q4 的 `score` 越小越好，完美预测为 `0`；排行榜必须对 q4 按 `score` 升序排列。
- `mAP50_95` 使用 IoU 阈值 `0.50, 0.55, ..., 0.95`。
- AP 计算使用 `ultralytics.utils.metrics.ap_per_class`，IoU 计算使用 `ultralytics.utils.metrics.box_iou`。

评分完成后按 `submission_id` 更新 `public.competition_prediction_submissions`：

```sql
update public.competition_prediction_submissions
set
  status = 'succeeded',
  score = $score,
  metrics = $metrics::jsonb,
  evaluated_at = now(),
  validation_error = null
where id = $submission_id;
```

## 环境变量

- 服务启动时会自动读取项目根目录 `.env`，真实环境变量优先级高于 `.env`。
- `DATABASE_URL`: PostgreSQL 连接字符串。支持 `postgresql://...`，也兼容常见的 `postgresql+asyncpg://...`。未配置时服务仍可运行，但会跳过数据库更新。
- `EVALUATOR_WORK_DIR`: 下载和解压临时目录，默认 `var/evaluator_jobs`。
- `EVALUATOR_MAX_DOWNLOAD_BYTES`: 最大下载体积，默认 250MB。
- `EVALUATOR_MAX_UNCOMPRESSED_BYTES`: 最大解压后体积，默认 750MB。
- `EVALUATOR_MAX_ZIP_MEMBERS`: zip 最大成员数，默认 20000。
- `EVALUATOR_DOWNLOAD_TIMEOUT_SECONDS`: 下载超时，默认 120 秒。
- `EVALUATOR_RETAIN_WORK_DIR`: 是否保留任务目录，默认保留。

## 历史指标重算

如果之前已经按 `mAP50_90` 评分，且 `EVALUATOR_WORK_DIR` 中仍保留历史任务目录，可以用脚本按新的 `mAP50_95` 重新计算并更新数据库。

先 dry-run 检查会重算哪些提交：

```bash
python3 scripts/recompute_map50_95.py
```

确认无误后写回 `score`、`metrics` 和 `evaluated_at`：

```bash
python3 scripts/recompute_map50_95.py --apply
```

脚本默认只处理 `metrics` 中仍包含 `mAP50_90` 的 `succeeded` 提交，并通过 `evaluator_job_id` 查找 `EVALUATOR_WORK_DIR/<job_id>/extracted` 或同目录下保留的 zip。若生产库里的赛题字段不是 `topic_id`，可用 `--topic-column <列名>` 指定。

## Docker 部署

镜像会包含服务代码和本地 q1-q4 验证集、测试集 ground truth。`.env` 不会被打包进镜像，`compose.yaml` 会在启动容器时注入 `.env`。

构建并启动：

```bash
docker compose up -d --build
```

查看日志：

```bash
docker compose logs -f evaluator
```

接口地址：

```text
http://localhost:8000/submissions
```

测试集评分接口：

```http
POST /test-submissions
Content-Type: application/json
```

```json
{
  "submission_id": "uuid-of-competition-test-prediction-submission",
  "topic_id": 1,
  "file_name": "prediction.zip",
  "file_size": 12345678,
  "download_url": "https://example.com/presigned-download-url"
}
```

- `topic_id=1` 使用 `q1_test/labels` 作为真值，计算 `mAP50`、`mAP50_95` 和 `score`。
- `topic_id=2` 使用 `q2_test/labels` 作为真值，计算 `mAP50_95` 和 `score`。
- `topic_id=3` 使用 `q3_test/test_result.xlsx` 作为真值，直接接收单个 `.xlsx`，独立计算 q3 `error`。
- `topic_id=4` 使用 `q4_test/test.txt` 作为真值，直接接收单个 `.txt`，独立计算 q4 `mse`。
- 这个接口和 `/submissions` 一样，通过同步校验后返回 `accepted`、`job_id` 和 `message`，后台评分完成后写入 `public.competition_test_prediction_submissions` 的 `status`、`score`、`metrics`、`evaluator_job_id`、`evaluated_at` 和 `validation_error`。

健康检查：

```text
http://localhost:8000/health
```

停止服务：

```bash
docker compose down
```

如果数据库跑在宿主机上，容器内不能使用 `localhost` 访问宿主机数据库。把 `.env` 里的主机名改成容器可访问地址，例如：

```env
DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:54322/postgres
```

如果数据库是独立生产数据库，直接使用该数据库的内网或公网连接地址。
