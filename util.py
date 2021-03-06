import numpy as np
import yaml

def good_shape(n, min_aspect=1.0, max_aspect=6, max_width=np.inf, max_height=np.inf, min_width=1, min_height=1, method='this is only here for backwards compatibility'):
    from numpy import log
    import scipy.optimize
    import itertools
    c = np.float32([1, .1]) # 0.1*logW + log_H
    A_ub = np.float32([[-1, -1],
                       [-1, 1],
                       [1, -1]])
    b = np.float32([-log(n), log(max_aspect), -log(min_aspect)])
    bounds = ((log(min_height), log(max_height)), (log(min_width), log(max_width)))
    res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds)
    H, W = np.exp(res.x).tolist()
    candidates = []
    for op_a, op_bb in itertools.product([np.ceil, np.floor], [np.ceil, np.floor]):
        W_p = op_a(W)
        H_p = op_bb(H)
        if W_p*H_p >= n: # we don't really check all the constraints for feasibility, just the area constraint
            candidates.append((H_p, (int(H_p), int(W_p)))) # Heuristic: accept the shortest (sort-of) feasible solution

    candidates = sorted(candidates)
    #assert len(candidates) > 0
    return candidates[0][1]

def build_imagegrid(image_list, n_rows, n_cols, highlight=None, margin=0.1):

    n, width, height = image_list.shape
    image_list /= np.max(image_list)
    #image_list /= np.max(image_list, axis=(1,2), keepdims=True) + 1e-3
    side = image_list.shape[1]
    margin = int(np.ceil(side * margin))
    image = np.ones((n_rows * width + (n_rows) * margin, n_cols * height + (n_cols) * margin))*0.3
    if highlight is not None:
        row, col = divmod(highlight, n_cols)
        image[row * (side + margin):row * (side + margin) + margin*2 + side,
        col * (side + margin):col * (side + margin) + margin*2 + side] = 1.0

    for row in range(n_rows):
        for col in range(n_cols):
            if row * n_cols + col >= n:
                break
            image[row * (side+margin) + margin:row * (side+margin) + margin + side,
                  col * (side+margin) + margin:col * (side+margin) + margin + side] = image_list[row * n_cols + col]
    return image

def default_config():
    return { 'servers' : {
        'local' :
            { 'address' : 'tcp://127.0.0.1:5560',
              'username' : 'admin',
              'password' : 'secret'
            }
        }
    }

def load_config():
    try:
        return yaml.load(open('config.yaml', 'r'))
    except Exception as e:
        config = default_config()
        yaml.dump(config, open('config.yaml', 'w'), default_flow_style=False)
        return config