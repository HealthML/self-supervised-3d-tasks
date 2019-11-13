import self_supervised_3d_tasks.train_and_eval as train_and_eval

def main():

    with open("self_supervised_3d_tasks/config/rotation/ukb3d.json", 'r') as f:
        args = json.load(f)
    train_and_eval(args)