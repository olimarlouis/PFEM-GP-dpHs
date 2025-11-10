import getfem as gf
import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky,cho_solve,cho_factor,LinAlgError,det,inv,eigvalsh
from scipy.integrate import odeint



class PFEM_GP_PHS_WAVE:
    def __init__(self,dx,dx_theta):
        self.best_h = None
        self.best_NLML = np.inf
        self.matrice_calculated = False
        self.perm_matrix_calucalted = False
        self.is_k_inv_build = False
        
        self.m = gf.Mesh('cartesian',np.arange(0,1.000000001,dx))
        self.m.set_region(10,self.m.outer_faces()[:,0])
        self.m.set_region(11,self.m.outer_faces()[:,1])
        self.mf_q = gf.MeshFem(self.m,1)
        self.mf_q.set_fem(gf.Fem('FEM_PK(1,1)'))

        self.mf_p = gf.MeshFem(self.m,1)
        self.mf_p.set_fem(gf.Fem('FEM_PK(1,1)'))

        
        self.mim = gf.MeshIm(self.m, 4)

        self.param = None



        self.m_theta = gf.Mesh('cartesian',np.arange(0,1.000000001,dx_theta))
        self.m_theta.set_region(10,self.m_theta.outer_faces()[:,0])
        self.m_theta.set_region(11,self.m_theta.outer_faces()[:,1])
        self.mf_theta_q = gf.MeshFem(self.m_theta,1)
        theta_q_element = 'FEM_PK(1,1)'
        self.mf_theta_q.set_fem(gf.Fem(theta_q_element))

        self.mf_theta_p = gf.MeshFem(self.m_theta,1)
        theta_p_element = 'FEM_PK(1,1)'
        self.mf_theta_p.set_fem(gf.Fem(theta_p_element))

        self.mim_theta = gf.MeshIm(self.m_theta, 4)

        self.Nq = self.mf_q.nbdof()
        self.Np = self.mf_p.nbdof()
        self.n = self.Np+self.Nq


        self.Nq_theta = self.mf_theta_q.nbdof()
        self.Np_theta = self.mf_theta_p.nbdof()
        

        self.md = gf.Model('real')
        self.md.add_fem_variable('q',mf=self.mf_q)
        self.md.add_fem_variable('p',mf=self.mf_p)

        self.md.add_fem_data('q2',mf=self.mf_q)
        self.md.add_fem_data('p2',mf=self.mf_p)

        self.md.add_fem_data('mq',mf=self.mf_q)
        self.md.add_fem_data('mp',mf=self.mf_p)

        self.md.add_fem_data('lq',mf=self.mf_q)
        self.md.add_fem_data('lp',mf=self.mf_p)


        self.mq_data = np.zeros(self.Nq_theta)
        self.mp_data = np.zeros(self.Np_theta)

        self.lq_data = np.zeros(self.Nq_theta)
        self.lp_data = np.zeros(self.Np_theta)



        self.md.add_fem_data('mq_true',mf=self.mf_theta_q)
        self.md.add_fem_data('mp_true',mf=self.mf_theta_p)


        self.md.add_fem_data('lq_true',mf=self.mf_theta_q)
        self.md.add_fem_data('lp_true',mf=self.mf_theta_p)



        self.calculate_permanente_matrix()


        for ii in range(self.Nq_theta):
            self.md.add_fem_data('dmq'+str(ii),mf=self.mf_q)
            self.md.add_fem_data('dlq'+str(ii),mf=self.mf_q)
            self.md.add_fem_data('dmq_true'+str(ii),mf=self.mf_theta_q)
            self.md.add_fem_data('dlq_true'+str(ii),mf=self.mf_theta_q)
            self.set_dmq(ii)
            self.set_dlq(ii)

        for ii in range(self.Np_theta):
            self.md.add_fem_data('dmp'+str(ii),mf=self.mf_p)
            self.md.add_fem_data('dlp'+str(ii),mf=self.mf_p)
            self.md.add_fem_data('dmp_true'+str(ii),mf=self.mf_theta_p)
            self.md.add_fem_data('dlp_true'+str(ii),mf=self.mf_theta_p)
            self.set_dmp(ii)
            self.set_dlp(ii)

            
        


        
        

        self.sigma_f = 1
        






    def set_alpha_q_from_fun(self, alpha_q):
        v = self.mf_q.eval('alpha_q(x)',globals(),locals())
        self.md.set_variable('q',v)

    def set_alpha_p_from_fun(self, alpha_p):
        v = self.mf_p.eval('alpha_p(x)',globals(),locals())
        self.md.set_variable('p',v)


    def set_rho_from_fun(self, expr):
        self.md.add_macro('rho',expr)

    def set_T_from_fun(self, expr):
        self.md.add_macro('T',expr)
        
    def set_c_from_fun(self, expr):
        self.md.add_macro('c',expr)

    def set_alpha_q(self, v):

        self.md.set_variable('q',v)

    def set_alpha_p(self, v):
        self.md.set_variable('p',v)

    def set_alpha_q2(self, v):

        self.md.set_variable('q2',v)

    def set_alpha_p2(self, v):
        self.md.set_variable('p2',v)
    

    def set_sigma_f(self, sigma_f):
        self.sigma_f = sigma_f

    def set_sigma_q(self, sigma_q):
        self.sigma_q = sigma_q
    
    def set_sigma_p(self, sigma_p):
        self.sigma_p = sigma_p

    



    def set_mq(self, v):
        self.md.set_variable('mq_true',v)
        inter = lambda x : gf.compute_interpolate_on(self.mf_theta_q,v,[x])[0]
        v2 = self.mf_q.eval("inter(x)",globals(),locals())
        self.md.set_variable('mq',v2)
        self.mq_data = v


    def set_mp(self, v):
        self.md.set_variable('mp_true',v)
        inter = lambda x : gf.compute_interpolate_on(self.mf_theta_p,v,[x])[0]
        v2 = self.mf_p.eval("inter(x)",globals(),locals())
        self.md.set_variable('mp',v2)
        self.mp_data = v


    def set_lq(self, v):
        self.md.set_variable('lq_true',v)
        inter = lambda x : gf.compute_interpolate_on(self.mf_theta_q,v,[x])[0]
        v2 = self.mf_q.eval("inter(x)",globals(),locals())
        self.md.set_variable('lq',v2)
        self.lq_data = v
        

    def set_lp(self, v):
        self.md.set_variable('lp_true',v)
        inter = lambda x : gf.compute_interpolate_on(self.mf_theta_p,v,[x])[0]
        v2 = self.mf_p.eval("inter(x)",globals(),locals())
        self.md.set_variable('lp',v2)
        self.lp_data = v





    def set_dmq(self, ii):
        v = np.zeros(self.Nq_theta)
        v[ii] = 1
        self.md.set_variable('dmq_true'+str(ii),v)
        inter = lambda x : gf.compute_interpolate_on(self.mf_theta_q,v,[x])[0]
        v2 = self.mf_q.eval("inter(x)",globals(),locals())
        self.md.set_variable('dmq'+str(ii),v2)


    def set_dmp(self, ii):
        v = np.zeros(self.Np_theta)
        v[ii] = 1
        self.md.set_variable('dmp_true'+str(ii),v)
        inter = lambda x : gf.compute_interpolate_on(self.mf_theta_p,v,[x])[0]
        v2 = self.mf_p.eval("inter(x)",globals(),locals())
        self.md.set_variable('dmp'+str(ii),v2)


    def set_dlq(self, ii):
        v = np.zeros(self.Nq_theta)
        v[ii] = 1
        self.md.set_variable('dlq_true'+str(ii),v)
        inter = lambda x : gf.compute_interpolate_on(self.mf_theta_q,v,[x])[0]
        v2 = self.mf_q.eval("inter(x)",globals(),locals())
        self.md.set_variable('dlq'+str(ii),v2)
        

    def set_dlp(self, ii):
        v = np.zeros(self.Np_theta)
        v[ii] = 1
        self.md.set_variable('dlp_true'+str(ii),v)
        inter = lambda x : gf.compute_interpolate_on(self.mf_theta_p,v,[x])[0]
        v2 = self.mf_p.eval("inter(x)",globals(),locals())
        self.md.set_variable('dlp'+str(ii),v2)

    

    def calculate_k(self):
        
        return self.sigma_f*np.exp(-0.5*(gf.asm_generic(self.mim,0,"pow((q-q2)*lq,2)",-1,self.md)+gf.asm_generic(self.mim,0,"pow((p-p2)*lp,2)",-1,self.md)))

        
    def calculate_dqk(self,ii):

        return -gf.asm_generic(self.mim,0,"pow((q-q2),2)*lq*dlq"+str(ii),-1,self.md)
        

    def calculate_dpk(self,ii):

        return -gf.asm_generic(self.mim,0,"pow((p-p2),2)*lp*dlp"+str(ii),-1,self.md)





    def calculate_Mmq(self):
        return gf.asm_generic(self.mim,2,"mq*Test_q*Test2_q",-1,self.md, "select_output",'q','q').full()
    
    def calculate_Mmp(self):
        return gf.asm_generic(self.mim,2,"mp*Test_p*Test2_p",-1,self.md, "select_output",'p','p').full()
    
    def calculate_Mkq(self):
        return gf.asm_generic(self.mim,2,"Test_q*pow(lq,2)*Test2_q",-1,self.md, "select_output",'q','q').full()

    def calculate_Mkp(self):
        return gf.asm_generic(self.mim,2,"Test_p*pow(lp,2)*Test2_p",-1,self.md, "select_output",'p','p').full()






    def calculate_dMmq(self,ii):
        return gf.asm_generic(self.mim,2,"dmq"+str(ii)+"*Test_q*Test2_q",-1,self.md, "select_output",'q','q').full()
    
    def calculate_dMmp(self,ii):
        return gf.asm_generic(self.mim,2,"dmp"+str(ii)+"*Test_p*Test2_p",-1,self.md, "select_output",'p','p').full()
    
    def calculate_dMkq(self,ii):
        return gf.asm_generic(self.mim,2,"Test_q*2*lq*dlq"+str(ii)+"*Test2_q",-1,self.md, "select_output",'q','q').full()


    def calculate_dMkp(self,ii):
        return gf.asm_generic(self.mim,2,"Test_p*2*lp*dlp"+str(ii)+"*Test2_p",-1,self.md, "select_output",'p','p').full()




    def calculate_Mq(self):
        return gf.asm_generic(self.mim,2,"Test_q*Test2_q",-1,self.md, "select_output",'q','q').full()


    def calculate_Mp(self):
        return gf.asm_generic(self.mim,2,"Test_p*Test2_p",-1,self.md, "select_output",'p','p').full()
    

    def calculate_BL(self):
        return gf.asm_generic(self.mim,1,"Test_p",10,self.md, "select_output",'p')
    

    def calculate_BR(self):
        return gf.asm_generic(self.mim,1,"Test_p",11,self.md, "select_output",'p')
    



    def calculate_true_D(self):
        return gf.asm_generic(self.mim,2,"Test_q*Grad(Test2_p)",-1,self.md, "select_output",'q','p').full()
    

    
    def calculate_constitutive_CR_Q(self):
        return gf.asm_generic(self.mim,2,"Test_q * Test2_q*T",-1,self.md, "select_output",'q','q').full()
    
    def calculate_constitutive_CR_QP(self):
        return gf.asm_generic(self.mim,2,"Test_q * Test2_p*c",-1,self.md, "select_output",'q','p').full()
    
    def calculate_constitutive_CR_P(self):
        return gf.asm_generic(self.mim,2,"Test_p * Test2_p*(1/rho)",-1,self.md, "select_output",'p','p').full()
    

    def calculate_permanente_matrix(self):
        if not self.perm_matrix_calucalted:
            self.Mq = self.calculate_Mq()
            self.Mp = self.calculate_Mp()
            self.BL = self.calculate_BL()
            self.BR = self.calculate_BR()
            


            self.Mq_inv = np.linalg.inv(self.Mq)
            self.Mp_inv = np.linalg.inv(self.Mp)

            self.M_inv = np.concatenate((np.concatenate((self.Mq_inv,np.zeros((self.Np,self.Nq))),axis=0),np.concatenate((np.zeros((self.Nq,self.Np)),self.Mp_inv),axis=0)),axis=1)
        

            


            self.perm_matrix_calucalted = True


    def calculate_true_matrix(self):
        self.D = self.calculate_true_D()
        self.CR_P = self.calculate_constitutive_CR_P()
        self.CR_Q = self.calculate_constitutive_CR_Q()
        self.CR_QP = self.calculate_constitutive_CR_QP()


    def update_all_matrix(self):
        if not self.matrice_calculated:
            self.calculate_permanente_matrix()
            self.Mmq = self.calculate_Mmq()
            self.Mmp = self.calculate_Mmp()
            self.Mkq = self.calculate_Mkq()
            self.Mkp = self.calculate_Mkp()
        

            self.set_cholesky_and_beta()


            self.matrice_calculated = True


    def update_all_dmatrice(self):
        self.dMmq =[]
        self.dMkq =[]
        for ii in range(self.Nq_theta):
            self.dMmq.append(self.calculate_dMmq(ii))
            self.dMkq.append(self.calculate_dMkq(ii))

        self.dMmp =[]
        self.dMkp =[]
        for ii in range(self.Np_theta):
            self.dMmp.append(self.calculate_dMmp(ii))
            self.dMkp.append(self.calculate_dMkp(ii))
           
            


  
            
        
    
    def calculate_qmean(self,alpha_q):
        return self.Mmq@alpha_q
    
    def calculate_pmean(self,alpha_p):
        return self.Mmp@alpha_p
    
    def calculate_mean(self,alpha_q,alpha_p):
        return np.concatenate((self.calculate_qmean(alpha_q),self.calculate_pmean(alpha_p)),axis=None)
    

    

    def calculate_kernel(self,alpha_q,alpha_p,alpha_q2,alpha_p2):
        self.set_alpha_q(alpha_q)
        self.set_alpha_p(alpha_p)
        self.set_alpha_q2(alpha_q2)
        self.set_alpha_p2(alpha_p2)

        k = self.calculate_k()
        Mkq_alpha = np.array([self.Mkq@(alpha_q-alpha_q2)]).transpose()
        kq = self.Mkq-Mkq_alpha@Mkq_alpha.transpose()



        Mkp_alpha = np.array([self.Mkp@(alpha_p-alpha_p2)]).transpose()
        kp = self.Mkp-Mkp_alpha@Mkp_alpha.transpose()

      
        kqp = -Mkp_alpha@Mkq_alpha.transpose()


        left = np.concatenate((kq,kqp),axis=0)
        right = np.concatenate((kqp.T,kp),axis=0)

        return k*np.concatenate((left,right),axis=1)
    






    def calculate_dqmean(self,alpha_q,ii):
        return self.dMmq[ii]@alpha_q
    
    def calculate_dpmean(self,alpha_p,ii):
        return self.dMmp[ii]@alpha_p 
    
    def calculate_dq_mean(self,alpha_q,alpha_p,ii):
        return np.concatenate((self.calculate_dqmean(alpha_q,ii),np.zeros(self.Np)),axis=None)
        
    
    def calculate_dp_mean(self,alpha_q,alpha_p,ii):
        return np.concatenate((np.zeros(self.Nq),self.calculate_dpmean(alpha_p,ii)),axis=None)
        
        


    def build_dqAlpha0(self,ii):
        dAlpha0 = np.zeros((self.Nt,self.Nq+self.Np))

        for i in range(self.Nt):
            dAlpha0[i] = -self.calculate_dq_mean(self.Alpha_q[i],self.Alpha_p[i],ii)
        
        return dAlpha0.flatten()



    def build_dpAlpha0(self,ii):
        dAlpha0 = np.zeros((self.Nt,self.Nq+self.Np))

        for i in range(self.Nt):
            dAlpha0[i] = -self.calculate_dp_mean(self.Alpha_q[i],self.Alpha_p[i],ii)
        
        return dAlpha0.flatten()







    

    def calculate_dqkernel(self,alpha_q,alpha_p,alpha_q2,alpha_p2,ii):
        self.set_alpha_q(alpha_q)
        self.set_alpha_p(alpha_p)
        self.set_alpha_q2(alpha_q2)
        self.set_alpha_p2(alpha_p2)

        k = self.calculate_k()
        Mkq_alpha = np.array([self.Mkq@(alpha_q-alpha_q2)]).transpose()
        kq = self.Mkq-Mkq_alpha@Mkq_alpha.transpose()



        Mkp_alpha = np.array([self.Mkp@(alpha_p-alpha_p2)]).transpose()
        kp = self.Mkp-Mkp_alpha@Mkp_alpha.transpose()

      
        kqp = -Mkp_alpha@Mkq_alpha.transpose()


        left = np.concatenate((kq,kqp),axis=0)
        right = np.concatenate((kqp.T,kp),axis=0)



        
        

        dMkq_alpha = np.array([self.dMkq[ii]@(alpha_q-alpha_q2)]).transpose()
        aaa = -dMkq_alpha@Mkq_alpha.transpose()
        dkq =self.dMkq[ii] + aaa + aaa.T

        dkp = np.zeros((self.Np,self.Np))
      
        dkqp = -Mkp_alpha@dMkq_alpha.transpose()

        dleft = np.concatenate((dkq,dkqp),axis=0)
        dright = np.concatenate((dkqp.T,dkp),axis=0)



        return self.calculate_dqk(ii)*k*np.concatenate((left,right),axis=1)+k*np.concatenate((dleft,dright),axis=1)


    def calculate_dpkernel(self,alpha_q,alpha_p,alpha_q2,alpha_p2,ii):
        self.set_alpha_q(alpha_q)
        self.set_alpha_p(alpha_p)
        self.set_alpha_q2(alpha_q2)
        self.set_alpha_p2(alpha_p2)

        k = self.calculate_k()
        Mkq_alpha = np.array([self.Mkq@(alpha_q-alpha_q2)]).transpose()
        kq = self.Mkq-Mkq_alpha@Mkq_alpha.transpose()



        Mkp_alpha = np.array([self.Mkp@(alpha_p-alpha_p2)]).transpose()
        kp = self.Mkp-Mkp_alpha@Mkp_alpha.transpose()

      
        kqp = -Mkp_alpha@Mkq_alpha.transpose()


        left = np.concatenate((kq,kqp),axis=0)
        right = np.concatenate((kqp.T,kp),axis=0)



        dkq = np.zeros((self.Nq,self.Nq))


        dMkp_alpha = np.array([self.dMkp[ii]@(alpha_p-alpha_p2)]).transpose()
        aaa = -dMkp_alpha@Mkp_alpha.transpose()
        dkp = self.dMkp[ii] + aaa + aaa.T

     
        
        
        

      
        dkqp = -dMkp_alpha@Mkq_alpha.transpose()

       


        dleft = np.concatenate((dkq,dkqp),axis=0)
        dright = np.concatenate((dkqp.T,dkp),axis=0)



        return self.calculate_dpk(ii)*k*np.concatenate((left,right),axis=1)+k*np.concatenate((dleft,dright),axis=1)




    def build_dqK_matrix(self,ii):
        n = self.Np + self.Nq
        N = self.Nt
        K = np.zeros((N*n,N*n))
        for i in range(N):
            for j in range(i,N):
                k = self.calculate_dqkernel(self.Alpha_q[i],self.Alpha_p[i],self.Alpha_q[j],self.Alpha_p[j],ii)
                if i!=j:
                    K[i*n:(i+1)*n,j*n:(j+1)*n] = k
                    K[j*n:(j+1)*n,i*n:(i+1)*n] = k
                else:
                    K[i*n:(i+1)*n,i*n:(i+1)*n] = k

        return K
    


    def build_dpK_matrix(self,ii):
        n = self.Np + self.Nq
        N = self.Nt
        K = np.zeros((N*n,N*n))
        for i in range(N):
            for j in range(i,N):
                k = self.calculate_dpkernel(self.Alpha_q[i],self.Alpha_p[i],self.Alpha_q[j],self.Alpha_p[j],ii)
                if i!=j:
                    K[i*n:(i+1)*n,j*n:(j+1)*n] = k
                    K[j*n:(j+1)*n,i*n:(i+1)*n] = k
                else:
                    K[i*n:(i+1)*n,i*n:(i+1)*n] = k

        return K





    def build_d_sigma_f_K_matrix(self):
        n = self.Np + self.Nq
        N = self.Nt
        K = np.zeros((N*n,N*n))
        
        for i in range(N):
            for j in range(i,N):
                k = self.calculate_kernel(self.Alpha_q[i],self.Alpha_p[i],self.Alpha_q[j],self.Alpha_p[j])/self.sigma_f
                if i!=j:
                    K[i*n:(i+1)*n,j*n:(j+1)*n] = k
                    K[j*n:(j+1)*n,i*n:(i+1)*n] = k
                else:
                    K[i*n:(i+1)*n,i*n:(i+1)*n] = k

        return K
    


    def build_d_sigma_q_K_matrix(self):
        n = self.Np + self.Nq
        N = self.Nt
        K = np.zeros((N*n,N*n))
        for i in range(N):       
            for j in range(self.Nq):
                    K[i*n+j,i*n+j] =  1

        return K
    

    def build_d_sigma_p_K_matrix(self):
        n = self.Np + self.Nq
        N = self.Nt
        K = np.zeros((N*n,N*n))
        for i in range(N):       
            for j in range(self.Np):
                    K[i*n+self.Nq+j,i*n+self.Nq+j] =  1

        return K

























    
    
    def set_alpha_data(self,Alpha_q,Alpha_p):
        self.Alpha_q = Alpha_q
        self.Alpha_p = Alpha_p
        self.Alpha = np.concatenate((Alpha_q,Alpha_p),axis=1)

        self.Nt = len(Alpha_q)
        self.N = self.Nt
        
    def set_e_data(self,eq,ep):
        res = np.zeros((self.Nt,self.Nq+self.Np))

        for i in range(self.Nt):
            res[i] = np.concatenate((self.Mq@eq[i],self.Mp@ep[i]))
        self.e = res





    


    def update_hyperparameter(self,h):
        if self.param is None:

            self.update_all_hyperparameter(h)
            self.matrice_calculated=False
            self.is_k_inv_build = False
            self.update_all_matrix()

        else:
            is_same = True
            for i in range(len(h)):
                if h[i] != self.param[i]:
                    is_same = False
            if not is_same:
                self.update_all_hyperparameter(h)
                self.matrice_calculated=False
                self.is_k_inv_build = False
                self.update_all_matrix()
                
        self.param = h
        



    def update_all_hyperparameter(self,h):
        nq = self.Nq_theta
        np = self.Np_theta
 
        
        self.set_mq(h[:nq])
        self.set_mp(h[nq:np+nq])
        self.set_lq(h[np+nq:2*nq+np])
        self.set_lp(h[2*nq+np:2*np+2*nq])

        # lqq = [0 for i in range(nq)]
        # lqq[1:nq-1] = h[0:nq-2]
        # self.set_lq(lqq)
        self.set_sigma_f(h[2*np+2*nq])
        self.set_sigma_q(h[2*np+2*nq+1])
        self.set_sigma_p(h[2*np+2*nq+1])

        






    def build_K_matrix(self):
        n = self.Np + self.Nq
        N = self.Nt
        K = np.zeros((N*n,N*n))
        Sigma = np.concatenate((np.concatenate((self.sigma_q*np.eye(self.Nq),np.zeros((self.Np,self.Nq))),axis=0),np.concatenate((np.zeros((self.Nq,self.Np)),self.sigma_p*np.eye(self.Np)),axis=0)),axis=1)
        for i in range(N):
            for j in range(i,N):
                k = self.calculate_kernel(self.Alpha_q[i],self.Alpha_p[i],self.Alpha_q[j],self.Alpha_p[j])
                if i!=j:
                    K[i*n:(i+1)*n,j*n:(j+1)*n] = k
                    K[j*n:(j+1)*n,i*n:(i+1)*n] = k
                else:
                    K[i*n:(i+1)*n,i*n:(i+1)*n] = k+ Sigma

        self.K = K



    def build_K_matrix_for_regression(self,alpha):
        n = self.Np + self.Nq
        N = self.Nt
        K = np.zeros((N*n,n))
        for i in range(N):
            
            k = self.calculate_kernel(self.Alpha_q[i],self.Alpha_p[i],alpha[:self.Nq],alpha[self.Nq:])
            
            K[i*n:(i+1)*n,:n] = k

        return K


    def build_Alpha0(self):
        Alpha0 = np.zeros((self.Nt,self.Nq+self.Np))

        for i in range(self.Nt):
            Alpha0[i] = self.e[i]-self.calculate_mean(self.Alpha_q[i],self.Alpha_p[i])
        
        self.Alpha0 = Alpha0.flatten()





    def set_cholesky_and_beta(self):
        self.build_Alpha0()
        self.build_K_matrix()
        try:
            self.L_K = cho_factor(self.K,lower=True)[0]
            self.beta =  cho_solve((self.L_K,True), self.Alpha0)


        except LinAlgError:
            #print("AAAAAAAAAA")
            self.L_K = None
            #self.Kinv = inv(self.K)
            #self.is_k_inv_build = True
            #self.beta = self.Kinv@self.Alpha0


    def calculate_K_inv(self):
        
        if not self.is_k_inv_build:
            self.Kinv = sp.csc_array(cho_solve((self.L_K,True), np.eye(len(self.L_K))))
            self.is_k_inv_build = True
        
 


    

    def NLML(self,h):
        self.update_hyperparameter(h)
        

        if self.L_K is None:
            NLML = np.inf
            #NLML = 0.5*self.Alpha0.T.dot(self.beta)+np.log(det(self.K))+self.n*self.Nt/ 2.0 * np.log(2.0 * np.pi)
        else:
            NLML = 0.5*self.Alpha0.T.dot(self.beta)+np.sum(np.log(np.diag(self.L_K)))+self.n*self.Nt/ 2.0 * np.log(2.0 * np.pi)

        print(NLML)
        if NLML < self.best_NLML:
            self.best_h = np.copy(h)
            self.best_NLML = NLML
        
        
        return NLML
        

    def grad_NLML(self,h):
        self.update_hyperparameter(h)
        self.update_all_dmatrice()
        
        if self.L_K is None:
            return np.array([np.inf for i in range(len(h))])
        self.calculate_K_inv()
        beta_K_inv = self.Kinv-np.outer(np.array(self.beta), np.array(self.beta))

        grad = np.zeros(len(h))
        kk = 0
        mq_activate = True
        mp_activate = True
        lq_activate = True
        lp_activate = True
        sigma_f_activate = True
        sigma_p_activate = False

        if mq_activate:
            for ii in range(self.Nq_theta):
                grad[kk] = 2*self.beta.T.dot(self.build_dqAlpha0(ii))
                kk+=1

        if mp_activate:
            for ii in range(self.Np_theta):
                grad[kk] = 2*self.beta.T.dot(self.build_dpAlpha0(ii))
                kk+=1


        if lq_activate:
            for ii in range(self.Nq_theta):
                grad[kk] = np.einsum('ij,ji->', beta_K_inv, self.build_dqK_matrix(ii))
                kk+=1

        if lp_activate:
            for ii in range(self.Np_theta):
                grad[kk] = np.einsum('ij,ji->', beta_K_inv, self.build_dpK_matrix(ii))
                kk+=1

        if sigma_f_activate:

            grad[kk] = np.einsum('ij,ji->', beta_K_inv, self.build_d_sigma_f_K_matrix())
            kk+=1

        if sigma_p_activate:
            grad[kk] = np.einsum('ij,ji->', beta_K_inv, self.build_d_sigma_q_K_matrix())
            kk+=1
            grad[kk] = np.einsum('ij,ji->', beta_K_inv, self.build_d_sigma_p_K_matrix())
            kk+=1
        
        else:
            grad[kk] = np.trace(beta_K_inv)

    
        return 0.5*grad
    


















    def predict_e(self,alpha):
        PHS_kernel_res = self.build_K_matrix_for_regression(alpha)
        e = self.calculate_mean(alpha[:self.Nq],alpha[self.Nq:]) + PHS_kernel_res.T.dot(self.beta)

        eq = e[:self.Nq]
        ep = e[self.Nq:]
            
        return np.concatenate([self.Mq_inv@eq ,self.Mp_inv@ep])


    def __ode_fun_gp(self,alpha, t, uL,uR):
            
            """
            Differential equation system function for odeint.
            """

            PHS_kernel_res = self.build_K_matrix_for_regression(alpha)
     

            
            e = self.calculate_mean(alpha[:self.Nq],alpha[self.Nq:]) + PHS_kernel_res.T.dot(self.beta)
            eq = e[:self.Nq]
            ep = e[self.Nq:]
            dalphadt = np.concatenate([self.Mq_inv@self.D@self.Mp_inv@ep,-self.Mp_inv@self.D.T@self.Mq_inv@eq - self.Mp_inv@self.BL*uL(t) + self.Mp_inv@self.BR*uR(t)])
            return dalphadt
    

    def __ode_fun_gp_true(self,alpha, t, uL,uR):
            
            """
            Differential equation system function for odeint.
            """

            dot_Alpha_q = self.Mq_inv@self.calculate_true_alpha_q_dot(alpha[:self.Nq],alpha[self.Nq:])
     
            dot_Alpha_p = self.Mp_inv@self.calculate_true_alpha_p_dot(alpha[:self.Nq],alpha[self.Nq:],uL(t),uR(t))

            
            dalphadt = np.concatenate([dot_Alpha_q,dot_Alpha_p])
            

            return dalphadt
    


    def pred_trajectory(self,t_span, alpha0, uL,uR):
            self.update_all_matrix()
            
            x_gp = odeint(self.__ode_fun_gp, alpha0, t_span, args=(uL,uR))

            

            return x_gp

    def pred_trajectory_true(self,t_span, alpha0, uL,uR):
            self.update_all_matrix()
            
            x_gp = odeint(self.__ode_fun_gp_true, alpha0, t_span, args=(uL,uR))

            return x_gp
    
    def calculate_true_alpha_q_dot(self,alphaq,alphap):
        self.calculate_permanente_matrix()
        self.calculate_true_matrix()
        return (self.D@self.Mp_inv).dot(self.CR_P.dot(alphap)+self.CR_QP.dot(alphaq))
    

    

    def calculate_true_alpha_p_dot(self,alphaq,alphap,uL,uR):
        self.calculate_permanente_matrix()
        self.calculate_true_matrix()
        return (-self.D.T@self.Mq_inv).dot(self.CR_Q.dot(alphaq)+self.CR_QP.T.dot(alphap)) - self.BL*uL + self.BR*uR
    



 
        
    
    



