Great — below are **official AWS docs and authoritative pages** that match each Q&A item from your study notes. I’ve grouped them by question number so you can jump straight to the source material.

---

# 1 — Metric retention for terminated EC2 / deleted ELB

* CloudWatch metric retention schedule (1s → 3 hours; 1m → 15 days; 5m → 63 days; 1h → 455 days). ([Amazon Web Services, Inc.][1])
* CloudWatch metric resolution (standard vs high-resolution). ([AWS Documentation][2])

# 2 — Standard vs High-Resolution metrics (defaults, granularity)

* High-resolution vs standard resolution explanation (1s granularity = high-res; default is 1m standard). ([AWS Documentation][2])
* Embedded Metric Format (EMF) showing `StorageResolution` for 1s high-res metrics. ([AWS Documentation][3])

# 3 — Increasing security of CloudTrail logs (MFA Delete on S3)

* S3 MFA Delete (how it works and how to configure versioned buckets). ([AWS Documentation][4])
* Note / guidance: using MFA Delete on the CloudTrail logging bucket is a recommended control (AWS best-practice guidance references). ([Cloudanix][5])

# 4 — CloudTrail defaults (event history vs trails, 90 days)

* CloudTrail event history is enabled by default (90 days viewable in console). ([AWS Documentation][6])
* CloudTrail trails (deliver to S3) must be created to persist logs beyond event history / for multi-region or aggregated delivery. ([AWS Documentation][7])

# 5 — CloudWatch vs CloudTrail responsibilities (monitoring vs auditing)

* CloudWatch overview (metrics, alarms, logs, health/monitoring). ([AWS Documentation][8])
* CloudTrail overview (audit of API calls / account activity). ([AWS Documentation][9])

# 6 — IAM permission(s) to read CloudWatch Logs Insights

* CloudWatch Logs API & permissions reference (includes `DescribeLogGroups`, other actions). Use `logs:DescribeLogGroups` to list/select groups for Insights. ([AWS Documentation][10])

# 7 — CloudWatch Logs retention (default and configurable)

* CloudWatch Logs retention defaults and how to change retention per log group (default = indefinite; retention options from 1 day up to years). ([AWS Documentation][11])

# 8 — Services that emit events to CloudWatch Events / EventBridge (RDS, EC2, Auto Scaling, CloudTrail, ELB)

* EventBridge (CloudWatch Events) service event reference / list of services that send events. ([AWS Documentation][12])
* Auto Scaling events via EventBridge. ([AWS Documentation][13])
* ELB events reference for EventBridge. ([AWS Documentation][14])

# 9 — CloudWatch Metric Streams output formats

* Metric Streams supported output formats: JSON, OpenTelemetry 0.7, OpenTelemetry 1.0. ([AWS Documentation][15])

# 10 — Correct CloudWatch use cases (metrics, alarms, logs)

* CloudWatch use cases / features summary (monitoring, metrics, logs, alarms). ([AWS Documentation][8])

---

If you want, I can:

* Turn these into a single PDF with the links embedded, or
* Create a **1-page printable cheat sheet** that includes each doc link and a one-line summary, or
* Produce a short quiz based on these links to test retention.

Which should I do next?

[1]: https://aws.amazon.com/blogs/aws/new-high-resolution-custom-metrics-and-alarms-for-amazon-cloudwatch/?utm_source=chatgpt.com "High-Resolution Custom Metrics and Alarms for Amazon CloudWatch"
[2]: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/cloudwatch_concepts.html?utm_source=chatgpt.com "Metrics concepts - Amazon CloudWatch"
[3]: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Embedded_Metric_Format_Specification.html?utm_source=chatgpt.com "Specification: Embedded metric format"
[4]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/MultiFactorAuthenticationDelete.html?utm_source=chatgpt.com "Configuring MFA delete - Amazon Simple Storage Service"
[5]: https://cloudanix.com/docs/aws/audit/cloudtrailmonitoring/rules/cloudtrail_mfa_delete?utm_source=chatgpt.com "CloudTrail Logging Bucket Should Use MFA Delete Feature"
[6]: https://docs.aws.amazon.com/awscloudtrail/latest/userguide/view-cloudtrail-events.html?utm_source=chatgpt.com "Working with CloudTrail event history - AWS Documentation"
[7]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/enable-cloudtrail-logging-for-s3.html?utm_source=chatgpt.com "Enabling CloudTrail event logging for S3 buckets and objects"
[8]: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/working_with_metrics.html?utm_source=chatgpt.com "Metrics in Amazon CloudWatch"
[9]: https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-user-guide.html?utm_source=chatgpt.com "What Is AWS CloudTrail? - AWS CloudTrail - AWS Documentation"
[10]: https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/permissions-reference-cwl.html?utm_source=chatgpt.com "CloudWatch Logs permissions reference"
[11]: https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/Working-with-log-groups-and-streams.html?utm_source=chatgpt.com "Working with log groups and log streams - Amazon CloudWatch Logs"
[12]: https://docs.aws.amazon.com/eventbridge/latest/ref/events.html?utm_source=chatgpt.com "Events - Amazon EventBridge - AWS Documentation"
[13]: https://docs.aws.amazon.com/eventbridge/latest/ref/events-ref-autoscaling.html?utm_source=chatgpt.com "Amazon EC2 Auto Scaling events - Amazon EventBridge"
[14]: https://docs.aws.amazon.com/eventbridge/latest/ref/events-ref-elb.html?utm_source=chatgpt.com "Elastic Load Balancing events - Amazon EventBridge"
[15]: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch-Metric-Streams.html?utm_source=chatgpt.com "Use metric streams - Amazon CloudWatch - AWS Documentation"
