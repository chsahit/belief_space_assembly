_time_in_sim = 0.0
_num_posteriors = 0


def reset_time():
    global _time_in_sim
    _time_in_sim = 0.0


def get_time():
    return _time_in_sim


def add_time(delta):
    global _time_in_sim
    _time_in_sim += delta


def reset_posteriors():
    global _num_posteriors
    _num_posteriors = 0


def get_posterior_count():
    return _num_posteriors


def add_posteriors(n):
    global _num_posteriors
    _num_posteriors += n
