data {
  // --- Scalars and summary vectors
  int<lower=1>              Ntrt;                   // Number of treatments
  int<lower=1>              Nstr;                   // Number of principal strata
  int<lower=1>              Ncat;                   // Number of strata observation categories (e.g. IH)
  int<lower=1>              Nx;                     // Number of covariate strata
  vector[Nx]                px;                     // P(X=x), proportion of patients with X=x
  
  // --- Individual level data
  int<lower=1>              Ni;                     // number of individual observations
  int<lower=1, upper=4>     cat[Ni];                // relapse by treatment category for principal strata
  int<lower=0, upper=1>     Di[Ni];                 // disability indicators (confirmed)
  int<lower=1, upper=Nx>    Xstr[Ni];               // Covariate stratum for each patient
  
  // --- Prior parameters
  vector[2]                 alpha_prior;            // alpha priors
  vector[2]                 alpha_H_prior;          // prior to use specifically for harmed stratum
  vector[2]                 theta_pbo_prior[Nstr];  // theta placebo priors
  vector[2]                 delta_prior[Nstr];      // delta prior parms
  
  // --- Utility data for run management, not used for modeling
  int<lower=0, upper=1>     ind_prior;              // indicator for th0 \perp th1 priors or th1 = th0 + delta priors
  int<lower=0, upper=1>     use_data;               // indicator as to whether to sample from prior or posterior
  
  // --- Various constant indices useful for code readability
  int                       IMM;
  int                       DOO;
  int                       BEN;
  int                       HAR;
  int                       DB;                     // doomed or benefiter
  int                       DH;                     // doomed or harmed
  int                       IB;                     // immune or benefiter
  int                       IH;                     // immune or harmed
  int                       PBO;
  int                       BAF;
  
}
parameters {
  // alpha parmeters refer to principal strata (relapse)
  vector[Nx] alpha_D;
  vector[Nx] alpha_I;
  vector[Nx] alpha_H;
  
  // theta and delta parameters refer to disability
  vector[Nx] theta_pbo[Nstr];
  vector[Nx] delta[Nstr];
  
} 
transformed parameters {
  vector[Nstr]              alpha[Nx];
  matrix[Nstr,Ntrt]         theta[Nx];
  
  for (x in 1:Nx) {
    alpha[x,IMM] = alpha_I[x];
    alpha[x,DOO] = alpha_D[x];
    alpha[x,HAR] = alpha_H[x];
    alpha[x,BEN] = 0;
    
    for (s in 1:Nstr) {
      theta[x,s,PBO] = theta_pbo[s,x];
      if (ind_prior == 1) {
        theta[x,s,BAF] = delta[s,x];
      } else {
        theta[x,s,BAF] = theta_pbo[s,x] + delta[s,x];
      }
    }
  }
} 
model {
  alpha_D ~ normal(alpha_prior[1],alpha_prior[2]);
  alpha_I ~ normal(alpha_prior[1],alpha_prior[2]);
  alpha_H ~ normal(alpha_H_prior[1],alpha_H_prior[2]);
  
  theta_pbo[IMM] ~ normal(theta_pbo_prior[IMM,1],theta_pbo_prior[IMM,2]);
  theta_pbo[DOO] ~ normal(theta_pbo_prior[DOO,1],theta_pbo_prior[DOO,2]);
  theta_pbo[BEN] ~ normal(theta_pbo_prior[BEN,1],theta_pbo_prior[BEN,2]);
  theta_pbo[HAR] ~ normal(theta_pbo_prior[HAR,1],theta_pbo_prior[HAR,2]);
  delta[IMM] ~ normal(delta_prior[IMM,1],delta_prior[IMM,2]);
  delta[DOO] ~ normal(delta_prior[DOO,1],delta_prior[DOO,2]);
  delta[BEN] ~ normal(delta_prior[BEN,1],delta_prior[BEN,2]);
  delta[HAR] ~ normal(delta_prior[HAR,1],delta_prior[HAR,2]);

  {
    int xi; 
    int ci;
    
    vector[Ni]     rprob;
    vector[Ni]     dprob;  
    vector[Ncat]   mix_weight[Nx];
    vector[Nstr]  pi_temp; 
    
    if (use_data == 1) {
      
      for (x in 1:Nx) {
        pi_temp = softmax(alpha[x]); 
        mix_weight[x,IH] = pi_temp[IMM]/(pi_temp[IMM]+pi_temp[HAR]);
        mix_weight[x,IB] = pi_temp[IMM]/(pi_temp[IMM]+pi_temp[BEN]);
        mix_weight[x,DH] = pi_temp[DOO]/(pi_temp[DOO]+pi_temp[HAR]);
        mix_weight[x,DB] = pi_temp[DOO]/(pi_temp[DOO]+pi_temp[BEN]);
      }
      
      for (i in 1:Ni) {
        ci = cat[i];
        xi = Xstr[i];
        
        if (ci==IH) { 
          // immune or harmed, placebo
          rprob[i] = log_sum_exp(categorical_logit_lpmf(IMM | alpha[xi]),
                                 categorical_logit_lpmf(HAR | alpha[xi]));
          dprob[i] = log_mix(mix_weight[xi,ci],
                             bernoulli_logit_lpmf(Di[i] | theta[xi,IMM,PBO]),
                             bernoulli_logit_lpmf(Di[i] | theta[xi,HAR,PBO]));
        } else if (ci==IB) { 
          // immune or benefiter, experimental
          rprob[i] = log_sum_exp(categorical_logit_lpmf(IMM | alpha[xi]),
                                 categorical_logit_lpmf(BEN | alpha[xi]));
          dprob[i] = log_mix(mix_weight[xi,ci],
                             bernoulli_logit_lpmf(Di[i] | theta[xi,IMM,BAF]),
                             bernoulli_logit_lpmf(Di[i] | theta[xi,BEN,BAF]));
        } else if (ci==DH) { 
          // doomed or harmed, experimental
          rprob[i] = log_sum_exp(categorical_logit_lpmf(DOO | alpha[xi]),
                                 categorical_logit_lpmf(HAR | alpha[xi]));
          dprob[i] = log_mix(mix_weight[xi,ci],
                             bernoulli_logit_lpmf(Di[i] | theta[xi,DOO,BAF]),
                             bernoulli_logit_lpmf(Di[i] | theta[xi,HAR,BAF]));                                 
        } else { 
          // doomed or benefiter, placebo
          rprob[i] = log_sum_exp(categorical_logit_lpmf(DOO | alpha[xi]),
                                 categorical_logit_lpmf(BEN | alpha[xi]));
          dprob[i] = log_mix(mix_weight[xi,ci],
                             bernoulli_logit_lpmf(Di[i] | theta[xi,DOO,PBO]),
                             bernoulli_logit_lpmf(Di[i] | theta[xi,BEN,PBO]));   
        }
      }
      target += rprob;
      target += dprob;
    }
  }
}
generated quantities {
  vector[Nstr]          th_PS[Ntrt]; // Disability prob. for the immune stratum
  simplex[Nstr]         pi;   // Principal strata probabilities
  vector[Nstr]          rr_PS; // relative risk for immune popn.
  vector[Ntrt]          th_O; // disability probability for the whole population
  real                  rr_O; // relative risk for the whole population
  
  // In here we implement standardization (average of model-predicted
  // probability weighted by covariate combination)
  {
    real          temp_pi[Nstr,Nx];
    real          temp_th[Ntrt,Nstr,Nx];
    real          temp_th_O[Nstr];
    vector[Nstr]  temp_sa;

    // For P(Stratum=s)
    for (s in 1:Nstr) {
      for (x in 1:Nx) {
        temp_sa = softmax(alpha[x]);
        // note temp_sa is covariate-specific, i.e. changes with x
        temp_pi[s,x] = temp_sa[s]*px[x]; 
      }
      pi[s] = sum(temp_pi[s]);
    }

    // For P(D(z) = 1 | S)
    for (z in 1:Ntrt) {
      for (s in 1:Nstr) {
        for (x in 1:Nx) {
          temp_th[z,s,x] = inv_logit(theta[x,s,z])*temp_pi[s,x];
        }
        th_PS[z,s] = sum(temp_th[z,s])/pi[s];
        temp_th_O[s] = th_PS[z,s]*pi[s];
      }
      th_O[z] = sum(temp_th_O);
    }

    for (s in 1:Nstr) {
      rr_PS[s] = th_PS[BAF,s]/th_PS[PBO,s];
    }
    rr_O = th_O[BAF]/th_O[PBO];
  }
}
