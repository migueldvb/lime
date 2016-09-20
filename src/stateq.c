/*
 *  stateq.c
 *  This file is part of LIME, the versatile line modeling engine
 *
 *  Copyright (C) 2006-2014 Christian Brinch
 *  Copyright (C) 2015-2016 The LIME development team
 *
 */

#include "lime.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>


void
stateq(int id, struct grid *g, molData *m, const int ispec, configInfo *par\
  , struct blendInfo blends, int nextMolWithBlend, gridPointData *mp\
  , double *halfFirstDs, _Bool *luWarningGiven){

  int t,s,iter,status;
  double *opop,*oopop,*tempNewPop=NULL;
  double diff;
  char errStr[80];

  gsl_matrix *matrix = gsl_matrix_alloc(m[ispec].nlev+1, m[ispec].nlev+1);
  gsl_matrix *reduc  = gsl_matrix_alloc(m[ispec].nlev, m[ispec].nlev);
  gsl_vector *newpop = gsl_vector_alloc(m[ispec].nlev);
  gsl_vector *rhVec  = gsl_vector_alloc(m[ispec].nlev);
  gsl_permutation *p = gsl_permutation_alloc (m[ispec].nlev);

  opop       = malloc(sizeof(*opop)      *m[ispec].nlev);
  oopop      = malloc(sizeof(*oopop)     *m[ispec].nlev);
  tempNewPop = malloc(sizeof(*tempNewPop)*m[ispec].nlev);

  for(t=0;t<m[ispec].nlev;t++){
    opop[t]=0.;
    oopop[t]=0.;
    gsl_vector_set(rhVec,t,0.);
  }
  gsl_vector_set(rhVec,m[ispec].nlev-1,1.);
  diff=1;
  iter=0;

  while((diff>TOL && iter<MAXITER) || iter<5){
    getjbar(id,m,g,ispec,par,blends,nextMolWithBlend,mp,halfFirstDs);

    getmatrix(id,matrix,m,g,ispec,mp);
    for(s=0;s<m[ispec].nlev;s++){
      for(t=0;t<m[ispec].nlev-1;t++){
        gsl_matrix_set(reduc,t,s,gsl_matrix_get(matrix,t,s));
      }
      gsl_matrix_set(reduc,m[ispec].nlev-1,s,1.);
    }

    status = gsl_linalg_LU_decomp(reduc,p,&s);
    if(status){
      if(!silent){
        sprintf(errStr, "LU decomposition failed for point %d, iteration %d (GSL error %d).", id, iter, status);
        bail_out(errStr);
      }
      exit(1);
    }

    status = gsl_linalg_LU_solve(reduc,p,rhVec,newpop);
    if(status){
      if(!silent && !(*luWarningGiven)){
        *luWarningGiven = 1;
        sprintf(errStr, "LU solver failed for point %d, iteration %d (GSL error %d).", id, iter, status);
        warning(errStr);
        warning("Doing LSE for this point. NOTE that no further warnings will be issued.");
      }
      lteOnePoint(m, ispec, g[id].t[0], tempNewPop);
      for(s=0;s<m[ispec].nlev;s++)
        gsl_vector_set(newpop,s,tempNewPop[s]);
    }

    diff=0.;
    for(t=0;t<m[ispec].nlev;t++){
      gsl_vector_set(newpop,t,gsl_max(gsl_vector_get(newpop,t),1e-30));
      oopop[t]=opop[t];
      opop[t]=g[id].mol[ispec].pops[t];

#pragma omp critical
      {
        g[id].mol[ispec].pops[t]=gsl_vector_get(newpop,t);
      }

      if(gsl_min(g[id].mol[ispec].pops[t],gsl_min(opop[t],oopop[t]))>minpop){
        diff=gsl_max(fabs(g[id].mol[ispec].pops[t]-opop[t])/g[id].mol[ispec].pops[t],gsl_max(fabs(g[id].mol[ispec].pops[t]-oopop[t])/g[id].mol[ispec].pops[t],diff));
      }
    }
    iter++;
  }

  gsl_matrix_free(matrix);
  gsl_matrix_free(reduc);
  gsl_vector_free(rhVec);
  gsl_vector_free(newpop);
  gsl_permutation_free(p);
  free(tempNewPop);
  free(opop);
  free(oopop);
}

double
e_temperature(double r){
  double Te;
  double rcs = 1.125e6; /* m scaling factors missing */
  double Tkin = 50;
  double Tmax = 1e4;
  if (r < rcs) {
    Te = Tkin;
  }
  else if (r > 2*rcs) {
    Te = Tmax;
  }
  else {
    Te = 40 + (Tmax - 40)*(r/rcs-1);
  }
  return Te;
}

void
getmatrix(int id, gsl_matrix *matrix, molData *m, struct grid *g, int ispec, gridPointData *mp){
  int ti,k,l,li,ipart,di,iline;
  double *girtot;
  struct getmatrix {
    double *ctot;
    gsl_matrix * colli;
  } *partner;
  /* IR pumping for ortho-water */
  /* double gir[7][7] = { */
  /* {0., 1.654e-5, 2.464e-5, 8.486e-5, 1.471e-4, 1.359e-5, 2.905e-5}, */
  /* {1.423e-5, 0., 1.882e-4, 1.696e-5, 1.118e-5, 9.143e-5, 1.294e-5}, */
  /* {1.323e-5, 1.154e-4, 0., 1.315e-5, 1.492e-5, 1.341e-4, 1.421e-5}, */
  /* {5.619e-5, 1.602e-5, 1.531e-5, 0., 2.846e-5, 1.694e-5, 1.415e-4}, */
  /* {6.620e-5, 5.189e-6, 1.113e-5, 1.982e-5, 0., 1.013e-5, 5.302e-5}, */
  /* {4.428e-6, 4.090e-5, 1.006e-4, 5.843e-6, 7.287e-6, 0., 9.177e-6}, */
  /* {1.448e-5, 7.867e-6, 9.250e-6, 1.038e-4, 5.593e-5, 1.166e-5, 0.}}; */
  /* double girtot[7] = {0}; */
  double gir[256][256]  = {0};

  girtot  = malloc(sizeof(double)*m[ispec].nlev);
  for(k=0;k<m[ispec].nlev;k++){
    girtot[k] = 0;
  }

  partner = malloc(sizeof(struct getmatrix)*m[ispec].npart);

  /* Initialize matrix with zeros */
  for(ipart=0;ipart<m[ispec].npart;ipart++){
    partner[ipart].colli = gsl_matrix_alloc(m[ispec].nlev+1,m[ispec].nlev+1);
    if(m[ispec].nlev>0) partner[ipart].ctot  = malloc(sizeof(double)*m[ispec].nlev);
    else {
      if(!silent)bail_out("Matrix initialization error in stateq");
      exit(0);
    }
    for(k=0;k<m[ispec].nlev+1;k++){
      for(l=0;l<m[ispec].nlev+1;l++){
        gsl_matrix_set(matrix, k, l, 0.);
        gsl_matrix_set(partner[ipart].colli, k, l, 0.);
      }
    }
  }

  /* Populate matrix with radiative transitions */
  for(li=0;li<m[ispec].nline;li++){
    k=m[ispec].lau[li];
    l=m[ispec].lal[li];
    gsl_matrix_set(matrix, k, k, gsl_matrix_get(matrix, k, k)+m[ispec].beinstu[li]*mp[ispec].jbar[li]+m[ispec].aeinst[li]);
    gsl_matrix_set(matrix, l, l, gsl_matrix_get(matrix, l, l)+m[ispec].beinstl[li]*mp[ispec].jbar[li]);
    gsl_matrix_set(matrix, k, l, gsl_matrix_get(matrix, k, l)-m[ispec].beinstl[li]*mp[ispec].jbar[li]);
    gsl_matrix_set(matrix, l, k, gsl_matrix_get(matrix, l, k)-m[ispec].beinstu[li]*mp[ispec].jbar[li]-m[ispec].aeinst[li]);
  }

  double distance = sqrt(g[id].x[0]* g[id].x[0]+g[id].x[1]*g[id].x[1]+g[id].x[2]*g[id].x[2]);
  /* Populate matrix with collisional transitions */
  for(ipart=0;ipart<m[ispec].npart;ipart++){
    struct cpData part = m[ispec].part[ipart];
    double *downrates = part.down;
    for(ti=0;ti<part.ntrans;ti++){
      int coeff_index = ti*part.ntemp + g[id].mol[ispec].partner[ipart].t_binlow;
      double down = downrates[coeff_index] + g[id].mol[ispec].partner[ipart].interp_coeff*(downrates[coeff_index+1] - downrates[coeff_index]);
      double up = down*m[ispec].gstat[part.lcu[ti]]/m[ispec].gstat[part.lcl[ti]]*exp(-HCKB*(m[ispec].eterm[part.lcu[ti]]-m[ispec].eterm[part.lcl[ti]])/g[id].t[0]);
      gsl_matrix_set(partner[ipart].colli, part.lcu[ti], part.lcl[ti], down);
      gsl_matrix_set(partner[ipart].colli, part.lcl[ti], part.lcu[ti], up);
    }

    /* collisions with electrons */
    if (ipart == 1) {
      for(iline=0;iline<m[0].nline;iline++){
        double aij = HPLANCK*m[0].freq[iline]/2./KBOLTZ/e_temperature(distance);
       double sigmaij = 9.10938356e-31*pow(1.6021766208e-19,2)*pow(299792458.0,3)*m[0].aeinst[iline]/16/pow(PI,2)/8.854187817620389e-12/pow(HPLANCK,2)/pow(m[0].freq[iline],4);
        double ve = sqrt(8*KBOLTZ*e_temperature(distance)/PI/9.10938356e-31);
       double bessel = gsl_sf_bessel_K0(aij);
        double ceij = ve*sigmaij*2.*aij*exp(aij)*bessel;
        double gij = m[0].gstat[m[0].lau[iline]]/m[0].gstat[m[0].lal[iline]];
        double ceji = ve*gij*sigmaij*2.*aij*exp(-aij)*bessel;
        gsl_matrix_set(partner[ipart].colli, m[0].lau[iline], m[0].lal[iline], ceij);
        gsl_matrix_set(partner[ipart].colli, m[0].lal[iline], m[0].lau[iline], ceji);
      }
    }

    for(k=0;k<m[ispec].nlev;k++){
      partner[ipart].ctot[k]=0.;
      for(l=0;l<m[ispec].nlev;l++)
        partner[ipart].ctot[k] += gsl_matrix_get(partner[ipart].colli,k,l);
    }
  }


  for(k=0;k<m[ispec].nlev;k++){
    for(l=0;l<m[ispec].nlev;l++)
      girtot[k] += gir[k][l];
  }

  for(k=0;k<m[ispec].nlev;k++){
    for(ipart=0;ipart<m[ispec].npart;ipart++){
      di = m[ispec].part[ipart].densityIndex;
      if(di>=0)
	 gsl_matrix_set(matrix,k,k,gsl_matrix_get(matrix,k,k)+g[id].dens[di]*partner[ipart].ctot[k]);
    }
    for(l=0;l<m[ispec].nlev;l++){
      if(k!=l){
        for(ipart=0;ipart<m[ispec].npart;ipart++){
          di = m[ispec].part[ipart].densityIndex;
          if(di>=0)
            gsl_matrix_set(matrix,k,l,gsl_matrix_get(matrix,k,l)-g[id].dens[di]*gsl_matrix_get(partner[ipart].colli,l,k));
        }
      }
    }
    gsl_matrix_set(matrix, m[ispec].nlev, k, 1.);
    gsl_matrix_set(matrix, k, m[ispec].nlev, 0.);
  }

  for(k=0;k<7;k++){
    gsl_matrix_set(matrix,k,k,gsl_matrix_get(matrix,k,k)+girtot[k]);
    for(l=0;l<7;l++){
      if(k!=l)
        gsl_matrix_set(matrix,k,l,gsl_matrix_get(matrix,k,l)-gir[l][k]);
    }
    gsl_matrix_set(matrix, m[ispec].nlev, k, 1.);
    gsl_matrix_set(matrix, k, m[ispec].nlev, 0.);
  }

  for(ipart=0;ipart<m[ispec].npart;ipart++){
    gsl_matrix_free(partner[ipart].colli);
    free(partner[ipart].ctot);
  }
  free(partner);
}


