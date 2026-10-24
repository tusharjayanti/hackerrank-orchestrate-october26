import os

# The test suite never calls the model, even when .env holds a key.
os.environ["GHOSTAUDIT_OFFLINE"] = "1"
