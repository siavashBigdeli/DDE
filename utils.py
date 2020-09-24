import numpy as np

def sample_2d_data(dataset, n_samples):

    z = np.random.normal(loc=0, scale=1, size=(n_samples,2))
    
    if dataset == '8gaussians':
        scale = 4
        sq2 = 1/np.sqrt(2)
        centers = [(1,0), (-1,0), (0,1), (0,-1), (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
        centers = np.array([(scale * x, scale * y) for x,y in centers])
        inds = np.random.random_integers(low=0, high=len(centers)-1, size=(n_samples,))
        x = sq2 * (0.5 * z + centers[inds])
        return x.astype(np.float32)

    elif dataset == '2spirals':
        u1 = np.random.uniform(low=0, high=1, size=(n_samples // 2))
        u2 = np.random.uniform(low=0, high=1, size=(n_samples // 2))
        u3 = np.random.uniform(low=0, high=1, size=(n_samples // 2))
        
        n = np.sqrt(u1) * 540 * (2 * np.pi) / 360
        d1x = - np.cos(n) * n + u2 * 0.5
        d1y =   np.sin(n) * n + u3 * 0.5
        x = np.concatenate([np.stack([ d1x,  d1y], axis=1),
                       np.stack([-d1x, -d1y], axis=1)], axis=0) / 3
        x = x + 0.1*z
        np.random.shuffle(x)
        return x.astype(np.float32)

    elif dataset == 'checkerboard':
        u1 = np.random.uniform(low=0, high=1, size=(n_samples))
        u2 = np.random.uniform(low=0, high=1, size=(n_samples))
        inds = np.random.random_integers(low=0, high=1, size=(n_samples,))
        
        x1 = u1 * 4 - 2
        x2_ = u2 - inds * 2
        x2 = x2_ + np.floor(x1) % 2
        x = np.stack([x1, x2], axis=1) * 2
        return x.astype(np.float32)
        

    elif dataset == 'rings':
        inds = np.random.random_integers(low=0, high=n_samples-1, size=(n_samples,))
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4 + 1)[:-1]
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3 + 1)[:-1]
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2 + 1)[:-1]
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1 + 1)[:-1]

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        x = np.stack([np.concatenate([circ4_x, circ3_x, circ2_x, circ1_x]),
                         np.concatenate([circ4_y, circ3_y, circ2_y, circ1_y])], axis=1) * 3.0

        # random sample
        x = x[inds]

        # Add noise
        x = x + np.random.normal(loc=0, scale=0.08, size=x.shape)
        return x.astype(np.float32)
