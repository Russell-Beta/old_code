/*********************************************************/
/*                       bp.c                            */
/*     　백 프로퍼게이션에 의한 신경망 학습              */
/* 사용방법                                              */
/*      $./bp < (학습 데이터 세트의 파일명)              */
/*오차의 추이와 학습결과가 되는 결합계수 등을 출력합니다 */
/*********************************************************/

/* 헤더 파일 포함 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* 기호 상수 정의 */
#define INPUTNO 6   /* 입력층 셀 수 */ 
#define HIDDENNO 4  /* 중간층 셀 수 */ 
#define ALPHA 1 /* 학습 계수*/ 
#define SEED 65535    /* 난수 시드 */ 
#define MAXINPUTNO 5000    /* 학습 데이터 최대개수 */ 
#define BIGNUM 1    /* 오차 초깃값 */ 
#define LIMIT 0.0001    /* 오차 상한값 */ 

/* 함수의 프로토타입 선언 */
double s(double u) ; /* 시그모이드 함수 */
void initwh(double wh[HIDDENNO][INPUTNO+1]) ;
                         /* 중간층 가중치 초기화 */
void initwo(double wo[HIDDENNO+1]) ;/* 출력층 가중치 초기화 */
double drnd(void) ;/* 난수 생성     */
void print(double wh[HIDDENNO][INPUTNO+1]
          ,double wo[HIDDENNO+1]) ; /* 결과 출력 */
double forward(double wh[HIDDENNO][INPUTNO+1]
         ,double wo[HIDDENNO+1],double hi[]
         ,double e[INPUTNO+1]) ; /* 순방향 계산 */
void olearn(double wo[HIDDENNO+1],double hi[]
         ,double e[INPUTNO+1],double o) ; /* 출력층 가중치 조정 */
int getdata(double e[][INPUTNO+1]) ; /* 학습데이터 읽어들이기 */
void hlearn(double wh[HIDDENNO][INPUTNO+1]
         ,double wo[HIDDENNO+1],double hi[]
         ,double e[INPUTNO+1],double o) ; /* 중간층 가중치 조정 */
/*******************/ 
/*    main()함수   */ 
/*******************/ 
int main()
{
 double wh[HIDDENNO][INPUTNO+1] ;/* 중간층 가중치 */
 double wo[HIDDENNO+1] ;/* 출력층 가중치 */
 double e[MAXINPUTNO][INPUTNO+1] ;/* 학습 데이터 세트 */
 double hi[HIDDENNO+1] ;/* 중간층 출력 */
 double o ;/* 출력 */
 double err=BIGNUM ;/* 오차 평가 */
 int i,j ;/* 반복 제어 */
 int n_of_e ;/* 학습 데이터 개수 */
 int count=0 ;/* 반복 횟수 카운터 */

 /* 난수 초기화 */
 //srand(SEED) ;
 srand((unsigned)time(NULL));

 /* 가중치 초기화 */
 initwh(wh) ;
 initwo(wo) ;
 print(wh,wo) ; 

 /* 학습 데이터 읽기 */
 n_of_e=getdata(e) ;
 printf("number of data:%d\n",n_of_e) ;

 /* 학습 */
 while(err>LIMIT){
  err=0.0 ;
  for(j=0;j<n_of_e;++j){
   /* 순방향 계산 */
   o=forward(wh,wo,hi,e[j]) ;
   /* 출력층 가중치 조정 */
   olearn(wo,hi,e[j],o) ;
   /* 중간층 가중치 조정 */
   hlearn(wh,wo,hi,e[j],o) ;
   /* 오차 적산*/
   err+=(o-e[j][INPUTNO])* (o - e[j][INPUTNO]);
  }
  ++count ;
  /* 오차 출력 */
  printf("%d\t %lf %lf\r",count,err, wo[0]) ;
  fflush(stdout);
  if (count == 10000000)
	  break;

 }/* 학습 종료 */

 /* 연결강도 출력 */
 print(wh,wo) ; 

 /* 학습 데이터에 */
 for(i=0;i<n_of_e;++i){
  printf("%d ",i) ;
  for(j=0;j<INPUTNO+1;++j)
   printf("%lf ",e[i][j]) ;
  o=forward(wh,wo,hi,e[i]) ;
  printf("%lf\n",o) ;
 }

 return 0 ;
}

/**********************/
/*  hlearn()함수      */
/* 중간층 가중치 학습 */
/**********************/
void hlearn(double wh[HIDDENNO][INPUTNO+1]
    ,double wo[HIDDENNO+1]
    ,double hi[],double e[INPUTNO+1],double o)
{
 int i,j ;/* 반복 제어 */
 double dj ;/* 중간층 가중치 계산에 이용 */

 for(j=0;j<HIDDENNO;++j){/*중간층의 각 셀j를 대상*/
  dj=hi[j]*(1-hi[j])*wo[j]*(e[INPUTNO]-o)*o*(1-o) ;
  for(i=0;i<INPUTNO;++i)/*i번째 가중치 처리*/
   wh[j][i]+=ALPHA*e[i]*dj ;
  wh[j][i]+=ALPHA*(-1.0)*dj ;/* 임계치 학습 */
 }
}


/**********************/
/*  getdata()함수     */
/*학습 데이터 읽기    */
/**********************/
int getdata(double e[][INPUTNO+1])
{
 int n_of_e=0 ;/* 데이터 세트 개수 */
 int j=0 ;/* 반복 제어 */

 /* 데이터 입력 */
 while(scanf("%lf",&e[n_of_e][j])!=EOF){
  ++ j ;
  if(j>INPUTNO){/* 다음 데이터 */
   j=0 ;
   ++n_of_e ;
  }
 }
 return n_of_e ;
}

/**********************/
/*  olearn()함수      */
/* 출력층 가중치 학습 */
/**********************/
void olearn(double wo[HIDDENNO+1]
    ,double hi[],double e[INPUTNO+1],double o)
{
 int i ;/* 반복 제어 */
 double d ;/* 가중치 계산에 이용 */

 d=(e[INPUTNO]-o)*o*(1-o) ;/* 오차 계산 */
 for(i=0;i<HIDDENNO;++i){
  wo[i]+=ALPHA*hi[i]*d ;/* 연결강도 학습*/
 }
 wo[i]+=ALPHA*(-1.0)*d ;/* 임계치 학습 */
 
} 


/**********************/
/*  forward()함수     */
/*  순방향 계산       */
/**********************/
double forward(double wh[HIDDENNO][INPUTNO+1]
 ,double wo[HIDDENNO+1],double hi[],double e[INPUTNO+1])
{
 int i,j ;/* 반복 제어 */
 double u ;/* 가중합 계산 */
 double o ;/* 출력 계산 */

 /* hi 계산 */
 for(i=0;i<HIDDENNO;++i){
  u=0 ;/* 가중합을 구한다 */
  for(j=0;j<INPUTNO;++j)
   u+=e[j]*wh[i][j] ; 
  u-=wh[i][j] ;/* 임계치 처리 */
  hi[i]=s(u) ;
 }
 /*출력o 계산 */
 o=0 ;
 for(i=0;i<HIDDENNO;++i)
  o+=hi[i]*wo[i] ;
 o-=wo[i] ;/* 임계치 처리 */
 
 return s(o) ;
} 

/**********************/
/*   print()함수      */
/*   결과 출력        */
/**********************/
void print(double wh[HIDDENNO][INPUTNO+1]
          ,double wo[HIDDENNO+1])
{
 int i,j ;/* 반복 제어 */

 for(i=0;i<HIDDENNO;++i)
  for(j=0;j<INPUTNO+1;++j)
   printf("%lf ",wh[i][j]) ;
 printf("\n") ;
 for(i=0;i<HIDDENNO+1;++i)
  printf("%lf ",wo[i]) ;
 printf("\n") ;
} 

/**********************/
/*    initwo()함수    */
/*중간층 가중치 초기화*/
/**********************/
void initwh(double wh[HIDDENNO][INPUTNO+1])
{
 int i,j ;/* 반복 제어 */

/*　난수에 의한 연결강조 결정 */ 
 for(i=0;i<HIDDENNO;++i)
  for(j=0;j<INPUTNO+1;++j)
   wh[i][j]=drnd() ;
} 

/**********************/
/*    initwo()함수    */
/*출력층 가중치 초기화*/
/**********************/
void initwo(double wo[HIDDENNO+1])
{
 int i ;/* 반복 제어 */

 for(i=0;i<HIDDENNO+1;++i)
   wo[i]=drnd() ;
} 

/*******************/
/* drnd()함수      */
/* 난수 생성       */
/*******************/
double drnd(void)
{
 double rndno ;/* 생성한 난수 */
 while((rndno=(double)rand()/RAND_MAX)==1.0) ;
 rndno=rndno*2-1 ;/* -1 ~ 1 사이의 난수 생성 */
 return rndno; 
}

/*******************/
/* s()함수         */
/* 시그모이드 함수
	대신 Relu		*/
/*******************/
double s(double u)
{
	return 1.0/(1.0+exp(-u)) ;
}
double r(double u)
{
	if (u <= 0)
		return 0;
	return u;
}
