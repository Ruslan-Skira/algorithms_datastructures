## Python AWS Lambda: Notebook Auth & Query Implementation

This guide provides a clean, corrected implementation of the Lambda function described. It handles **Bearer Token** authentication via DynamoDB and retrieves the top 10 most recent notes for a validated user.

---

### **The Problem Description**

**Task:** Build an AWS Lambda (Python 3.8) that acts as an API Gateway backend for a notebook application.

**Requirements:**

1. **Authentication:**
* Extract a Bearer token from the `Authentication` header.
* Lookup the email associated with the token in the `token-email-lookup` table.
* **Error 400:** Header is missing or doesn't follow the `Bearer <token>` format.
* **Error 403:** Token is an empty string or not found in the database.


2. **Data Retrieval:**
* Query the `user-notes` table using the user's email.
* Return notes sorted by `create_date` in **descending** order.
* Limit the result to **10** items.


3. **Response:**
* Return a structured JSON response with appropriate HTTP status codes.



---

### **Implementation (Ready to Copy-Paste)**

```python
import json
import re
import boto3
from boto3.dynamodb.conditions import Key

class InvalidResponse(Exception):
    def __init__(self, status_code):
        self.status_code = status_code

def query_user_notes(user_email):
    """
    Queries user-notes table:
    - Partition Key: 'user'
    - Sort Key: 'create_date'
    - Returns: Top 10 latest notes (Descending)
    """
    dynamo_db = boto3.resource('dynamodb')
    user_notes_table = dynamo_db.Table('user-notes')

    # ScanIndexForward=False results in Descending order (latest first)
    result = user_notes_table.query(
        KeyConditionExpression=Key('user').eq(user_email),
        ScanIndexForward=False,
        Limit=10
    )
    return result.get('Items', [])

def get_authenticated_user_email(token):
    """
    Looks up the email associated with the provided token.
    Throws 403 if token is invalid or not found.
    """
    dynamo_db = boto3.resource('dynamodb')
    tokens_table = dynamo_db.Table('token-email-lookup')

    response = tokens_table.get_item(
        Key={'token': token}
    )

    if 'Item' not in response:
        # Token not found in database
        raise InvalidResponse(403)

    return response['Item']['email']

def authenticate_user(headers):
    """
    Validates the Authentication header format and token content.
    - 400 if header missing or malformed.
    - 403 if token is empty string.
    """
    # 1. Check if header exists
    if 'Authentication' not in headers:
        raise InvalidResponse(400)

    auth_header = headers['Authentication']

    # 2. Regex to handle Bearer format.
    # Using \s* allows matching even if the token part is empty/missing
    match = re.match(r'^Bearer\s*(.*)$', auth_header)
    if not match:
        raise InvalidResponse(400)

    token = match.group(1).strip()

    # 3. Requirement: 403 if token is empty
    if not token:
        raise InvalidResponse(403)

    user_email = get_authenticated_user_email(token)
    return user_email

def build_response(status_code, body=None):
    """
    Standard API Gateway response formatter.
    """
    result = {
        'statusCode': status_code, # Ensure integer for API Gateway
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }

    if body is not None:
        result['body'] = body
    return result

def handler(event: dict, context):
    """
    Main Lambda entry point.
    """
    try:
        # Extract headers (API Gateway passes them in event['headers'])
        headers = event.get('headers', {})
        user_email = authenticate_user(headers)

        # Query data
        notes = query_user_notes(user_email)

        return build_response(status_code=200, body=json.dumps(notes))

    except InvalidResponse as e:
        return build_response(status_code=e.status_code)
    except Exception:
        # Generic fallback for unexpected errors
        return build_response(status_code=500)

```

---

### **Learning Points**

#### **1. DynamoDB Query Parameters**

* **`KeyConditionExpression`**: Necessary for efficient querying by Partition Key.
* **`ScanIndexForward=False`**: This is the specific way to reverse the sort order in DynamoDB (default is Ascending).
* **`Limit=10`**: Restricts the number of items returned, saving read capacity units (RCU).

#### **2. The Regex Logic**

* Using `r'^Bearer\s*(.*)$'` ensures that strings like `"Bearer "` (with a trailing space but no token) still match the regex pattern.
* This allows the code to proceed to the `if not token:` check, which specifically satisfies the requirement to return a **403** instead of a **400** for empty tokens.

#### **3. API Gateway Response Format**

* The `statusCode` must be an integer (e.g., `200`, not `"200"`) for many Lambda integrations to work correctly without errors.
* `Access-Control-Allow-Origin: '*'` is included in the headers to prevent CORS issues when called from a browser-based frontend.

**Would you like me to create a mock `event` dictionary so you can run and test this code locally?**