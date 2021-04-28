NUM_PARALLEL_PROCESSORS=1
testNames=(offline romhr)
case $subTestNum in
  1)
      $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -offline -writesol -romsvds -no-romgs -no-romoffset -no-romsns -romst gnat_lspg
      ;;
  2)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -online -romhrprep -rdimx 10 -rdimv 10 -rdime 10 -rdimfv 15 -rdimfe 15 -ntsamv 10 -ntsame 10 -sfacv 10 -sface 10 -soldiff -no-romgs -no-romoffset -no-romsns -romst gnat_lspg
    $LAGHOS_SERIAL -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -online -romhr -rdimx 10 -rdimv 10 -rdime 10 -rdimfv 15 -rdimfe 15 -ntsamv 10 -ntsame 10 -sfacv 10 -sface 10 -soldiff -no-romgs -no-romoffset -no-romsns -romst gnat_lspg
    ;;
esac
