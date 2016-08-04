import numpy as np
import pdb
#from dtree import *
import model

def main():
   # detree = DTree(np.zeros(3), [1., 2., 3.])
   # 
   # inputs = [0.2, 0.6, 0.1]

   # print detree.infer(inputs)
   # detree.feedback(inputs, 1)
   # pdb.set_trace()
   # cuts = detree.save()
   # print cuts
   # detree.feedback(inputs, 0)
   # print detree.infer(inputs)
   testmod = model.Model(12)
   testmod.train()


if __name__ == '__main__':
    main()
