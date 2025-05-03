import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_last_accuracy_from_tensorboard(logdir, tag="metric/Accuracy"):
    try:
        ea = EventAccumulator(logdir)
        ea.Reload()
        if tag not in ea.Tags()['scalars']:
            return np.nan
        events = ea.Scalars(tag)
        if not events:
            return np.nan
        return events[-1].value * 100  # laatste accuracy in %
    except Exception as e:
        print(f"Fout bij {logdir}: {e}")
        return np.nan
    

def inspect_tensorboard_tags(logdir):
    ea = EventAccumulator(logdir)
    ea.Reload()
    print("Beschikbare scalar tags:", ea.Tags()['scalars'])