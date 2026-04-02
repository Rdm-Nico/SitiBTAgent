# How to run live test/test files
To run the live test files (live_test_w_*.py and all the other file a part of test_whatsapp_agent.py) you can run it like a module in the project root:
 ```bash
 # go in the project root
 cd .. 
 # run like a moduel 
 python -m test.live_test_w_vllm
```
Is important to not include the  '.py' at the end. 
The other file (test_whatsapp_agent.py) is a unittest file and it can run from the root like every unittest file