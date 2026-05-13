      SUBROUTINE TSYSTEM
      COMMON /BLOC1/vane,rfp,ax,bxu,exu,ay,byu,eyu,az,bzu,ezu
      COMMON/SCPARM/QSC,ISC,CMPS
      COMMON/MOM/P,BRHO,pMASS,energk,gsq,ENERGKi,charge,current
      COMMON/PRINT/IPRINT,IQ(8)

      REAL qenergk
      INTEGER nscav

      iq(1)=6

      CMPS=0.005
      nscav=100
      endDrift=1.0
      rmsDrift = 1.0

      OPEN(99,FILE='fort.energy',STATUS='unknown')

      call cic3(ax,bxu,exu,ay,byu,eyu,az,bzu,ezu)
      call vective(1)
      WRITE(99,'(A,1X,E16.8)') 'AFTER_SETUP', energk

      call dr(rmsDrift,".")
      WRITE(99,'(A,1X,E16.8)') 'AFTER_RMSDRIFT', energk

      call rfq(75,1273,vane,2.1403549619E+01,7.1400000000E+08,rfp,nscav)

      WRITE(99,'(A,1X,E16.8)') 'AFTER_RFQ', energk

      call dr(endDrift,".")
      WRITE(99,'(A,1X,E16.8)') 'AFTER_ENDDRIFT', energk

      CLOSE(99)

      qenergk=0.30
      call fitarb(0.0,qenergk-energk,10.,1)
      call fit(1,6,5,0.0,1.,1)

      return
      end
