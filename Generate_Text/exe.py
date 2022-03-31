# Auto generated code by 'nbox jobs new' command
# project name: Generate_Text
# created time: 2022-03-30 05:56:27 UTC
#   created by: 
#
# > feeling stuck, start by populating the functions below <
# all the excess code will be removed in future patch updates

import os
import sys
import requests
from fire import Fire
from text import *


# save the ~/.nbx/secrets.json
r = requests.get("https://raw.githubusercontent.com/NimbleBoxAI/nbox/master/assets/sample_config.json")
home_dir = os.path.join(os.path.expanduser("~"), ".nbx")
secrets = os.path.join(home_dir, "secrets.json")
os.makedirs(home_dir, exist_ok=True)
if not os.path.exists(secrets):
  with open(secrets, "w") as f:
    f.write(r.content.decode())

os.environ["NBOX_JOB_FOLDER"] = os.getcwd() # Do not touch
os.environ["NBOX_LOG_LEVEL"] = "INFO" # Keep it the way you like

import nbox.utils as U
from nbox.nbxlib.tracer import Tracer
from nbox import Operator, nbox_grpc_stub
from nbox.hyperloop.nbox_ws_pb2 import UpdateRunRequest
from nbox.hyperloop.job_pb2 import Job
from nbox.messages import rpc


def get_op() -> Operator:
  # since initialising your operator might require passing a bunch of arguments
  # you can use this function to get the operator by manually defining things here
    parent = "./data"
    text_names = []
    for item in os.listdir(parent):
        if item.endswith(".txt"):
            text_names.append(f"{item}")
    op = Text(parent, text_names)
    return op


def deploy():
  op: Operator = get_op()
  job = op.deploy(
    job_id_or_name = 'Generate_Text',

    init_folder = U.folder(__file__), # ! ~ do not change this
  )


def run():
  op: Operator = get_op()
  op.propagate(_tracer = Tracer())
  if hasattr(op._tracer, "job_proto"):
    op.thaw(op._tracer.job_proto)
  try:
    op(
      # your operator is going to run once, try passing all inputs here
    )
  except Exception as e:
    U.logger.error(e)
    if hasattr(op._tracer, "job_proto"):
      op._tracer.job_proto.status = Job.Status.ERROR
  else:
    if hasattr(op._tracer, "job_proto"):
      op._tracer.job_proto.status = Job.Status.COMPLETED
      rpc(
        nbox_grpc_stub.UpdateRun, UpdateRunRequest(token = op._tracer.token, job=op._tracer.job_proto), "Failed to end job!"
      )

if __name__ == "__main__":
  Fire({"deploy": deploy, "run": run})
  sys.exit(0)

# end of auto generated code