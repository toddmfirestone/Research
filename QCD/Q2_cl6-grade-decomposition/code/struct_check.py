import numpy as np, glob
for f in sorted(glob.glob('*.np[yz]')):
    try:
        o = np.load(f, allow_pickle=True)
        if hasattr(o, 'files'):
            print(f, '[npz]', {k: np.asarray(o[k]).shape for k in o.files})
        else:
            print(f, o.shape, o.dtype)
    except Exception as e:
        print(f, 'ERR', e)