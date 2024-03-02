import numpy as np

class LinSVM:
    def __init__(self, C = 1, eps = 1e-2, shrink_coeff = 1, max_iter = np.inf, store_primal = False):
        self.C = C
        self.eps = eps
        self.shrink_coeff = shrink_coeff
        self.w = None  #Vector w needs to be fitted to given data. Last element of w represents bias b.
        self.loss = list()
        self.accuracy = list()
        self.store_primal = store_primal
        self.max_iter = max_iter


    def fit(self, x: np.ndarray, y: np.ndarray, verbose = False):
        assert len(x.shape) == 2, f"Data matrix should be 2-dimensional!"
        assert len(y.shape) == 1, f"Labels array should be 1-dimensional!"
        assert x.shape[0] == y.shape[0], f"Number of data points ({x.shape[0]}) does not equal the number of labels ({y.shape[0]})!"

        num_opt= x.shape[0]  #The number of alphas to optimize is equal to the number of data points, it will get smaller after shrinking

        x = np.c_[x, np.ones(x.shape[0])]

        PG_max = np.inf
        PG_min = -np.inf


        alpha = np.full(num_opt, self.C/2)
        is_opt = np.full(num_opt, True)
        ay = alpha * y

        w = np.sum((x.T * ay).T, axis = 0)  #w = sum(ai * yi * xi)
        Q = np.sum(np.abs(x)**2, axis = 1)


        iter = 1
        while(PG_max - PG_min >= self.eps and iter <= self.max_iter):
            order = np.random.permutation(num_opt)

            new_PG_max = -np.inf
            new_PG_min = np.inf

            new_opt = is_opt.copy()

            where = np.where(is_opt)[0]

            for i in range(num_opt):
                
                current_idx = where[order[i]]
                current_alpha = alpha[current_idx]
                
                grad = y[current_idx] * np.dot(w, x[current_idx, :]) - 1
                skip = False


                if (current_alpha > 0 and  current_alpha < self.C):
                    proj_grad = grad

                elif current_alpha == 0:
                    proj_grad = min(grad, 0)
    
                    if (grad > self.shrink_coeff * PG_max):
                        new_opt[current_idx] = False
                        num_opt -= 1

                        skip = True
                        
                elif current_alpha == self.C:
                    proj_grad = max(grad, 0) 

                    if (grad < self.shrink_coeff * PG_min):
                        new_opt[current_idx] = False
                        num_opt -= 1

                        skip = True
                    

                if(proj_grad > new_PG_max):
                    new_PG_max = proj_grad
                if(proj_grad < new_PG_min):
                    new_PG_min = proj_grad

                if(proj_grad != 0 and not skip):
                    new_alpha = min(max(current_alpha - proj_grad/Q[current_idx], 0), self.C)
                    w = w + (new_alpha - current_alpha)*y[current_idx]*x[current_idx,:]
                    alpha[current_idx] = new_alpha
                    

            PG_max = new_PG_max
            PG_min = new_PG_min
            is_opt = new_opt.copy()

            self.w = w


            if self.store_primal:
                self.loss.append(self.primal_loss(x, y))

            if verbose:
                print(f"Iteration: {iter}")
                print(f"Gradient gap: {PG_max - PG_min}\n")
                iter = iter + 1

        

    def predict(self, new_x: np.ndarray):
        assert len(new_x.shape) == 2, f"Data matrix should be 2-dimensional"

        new_x = np.c_[new_x, np.ones(new_x.shape[0])]

        value = (new_x * self.w)
        value = np.sum(value, axis = 1)

        return(np.sign(value))
    

    def primal_loss(self, x, y):
        wtw = np.linalg.norm(self.w)**2
        violations = np.full(len(y), 1) - y * np.sum(x * self.w, axis = 1)
        penalty = np.sum(np.maximum(violations, np.full(len(y), 0)))

        return(1/2 * wtw - self.C * penalty)
    
