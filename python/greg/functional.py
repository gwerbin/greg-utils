from boltons.typeutils import make_sentinel


NOMATCH = make_sentinel('NOMATCH', 'NOMATCH')


def match1(*matches, name=None, default=NOMATCH):
    """ Pattern-matching for 1-argument functions
    
    match1(
       (lambda x: x > 5,      'high'),
       (lambda x: 2 < x <= 5, 'medium'),
       (lambda x: x <= 2,     'low'),
    )(3)
    """
    def matching_function(x):
        for condition, action in matches:
            do_action = (callable(condition) and condition(x)) or x == condition
            if do_action:
                if callable(action):
                    return action(x)
                else:
                    return action
        else:
            if default is NOMATCH:
                raise ValueError('No match found')
            else:
                return default

    if name:
        matching_function.__name__ = name

    return matching_function

