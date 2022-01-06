""" Class for evaluating a tempo tracker """
import math


class TrackerEvaluator:
    def __init__(self):
        pass

    def evaluate(self, tracker_class, X_train, y_train, **kwargs):
        preds = []
        for i, x in enumerate(X_train):
            tempo_tracker = tracker_class(**kwargs)
            for onset in x:
                tempo_tracker.run(onset)
            # Get last tempo estimate in BPM
            last_tempo_estimate = tempo_tracker.get_tempo_estimates()[-1]
            last_tempo_estimate_bpm = tempo_tracker.tempo_period_to_bpm(last_tempo_estimate)

            # Round tempo estimate to closest integer
            rounded_tempo = math.ceil(last_tempo_estimate_bpm) \
                if last_tempo_estimate_bpm % 1 >= 0.5 else int(last_tempo_estimate_bpm)

            preds.append(rounded_tempo)

        accuracy2 = self.get_accuracy2(preds, y_train)

        return accuracy2

    def get_accuracy2(self, preds, y_train, tolerance_percentage=4.0):
        accurate_preds = 0
        for i, pred in enumerate(preds):
            valid_tempos = [pred, pred * 2, pred * 3]
            ground_truth = y_train[i]
            lower_bound = ground_truth - ground_truth * tolerance_percentage / 100.0
            upper_bound = ground_truth + ground_truth * tolerance_percentage / 100.0
            for valid_tempo in valid_tempos:
                if lower_bound < valid_tempo < upper_bound:
                    accurate_preds += 1
                    break
        return accurate_preds / len(y_train)
