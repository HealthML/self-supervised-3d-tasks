import self_supervised_3d_tasks.train_and_eval as train_and_eval

def main():

    with open("self_supervised_3d_tasks/config/exemplar/ukb3d.sh", 'r') as f:
        args = json.load(f)
    train_and_eval(args)