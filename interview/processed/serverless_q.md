## Serverless Framework Deployment Configuration

This guide provides the `serverless.yml` configuration and a breakdown of the requirements for deploying an AWS Lambda function using the Serverless Framework (v1.x).

---

### **The Problem Description**

**Task:** Create a `serverless.yml` deployment file for a Python 3.8 application.

**Requirements:**

1. **Service Identity:** Name the service `helloer-service`.
2. **Version Control:** Restrict the framework version to be  and .
3. **Environment:** * **Region:** `eu-central-1` (Frankfurt).
* **Stage:** Override the default to `dev-env`.


4. **Function Definition:**
* **Name:** `helloer`.
* **Handler:** Located in `helooer.py` as a function named `hello_handler`.
* **Trigger:** An HTTP `GET` request to the `/hello` path.
* **Security:** Explicitly disable CORS.



---

### **Implementation (`serverless.yml`)**

```yaml
service: helloer-service

# Requirement: Framework version above 1.5.0 and below 2.0.0
frameworkVersion: ">=1.5.0 <2.0.0"

provider:
  name: aws
  runtime: python3.8
  # Requirement: Default region eu-central-1
  region: eu-central-1
  # Requirement: Override default stage with dev-env
  stage: dev-env

functions:
  helloer: # The function name defined in the deployment
    handler: helooer.hello_handler # [filename].[function_name]
    events:
      - http:
          path: /hello
          method: get
          # Requirement: CORS should be 'disabled'
          cors: false

```

---

### **Learning Points**

#### **1. Version Constraints**

The `frameworkVersion` property uses semantic versioning. Setting it to `">=1.5.0 <2.0.0"` ensures that if a teammate tries to deploy with a modern version (like v3.0), the deployment will stop, preventing potential syntax incompatibilities.

#### **2. Handler Resolution**

The framework looks for the code based on the `handler` string. In this case:

* **File:** `helooer.py`
* **Function:** `hello_handler`

#### **3. API Gateway Mapping**

When the `http` event is defined, the Serverless Framework automatically provisions an **AWS API Gateway**.

* The `path` maps to the URL.
* The `method` restricts the allowed HTTP verb.
* `cors: false` ensures that no `OPTIONS` pre-flight endpoint is created and no CORS headers (like `Access-Control-Allow-Origin`) are returned by the gateway.

#### **4. Service vs. Function Names**

* **Service Name (`helloer-service`):** This acts as the prefix for all AWS resources created (S3 buckets, CloudFormation stacks).
* **Function Key (`helloer`):** This is the internal name used within the YAML file and the specific name for the Lambda resource in the AWS Console.

---

### **Project File Structure**

To deploy this successfully, your project directory must contain these two files:

```text
.
├── helooer.py      # Contains: def hello_handler(event, context):
└── serverless.yml  # The configuration provided above

```

**Would you like me to provide a sample `helooer.py` file that specifically extracts and logs the API Gateway event data?**