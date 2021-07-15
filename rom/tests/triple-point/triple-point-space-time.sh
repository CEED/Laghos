NUM_PARALLEL_PROCESSORS=1
testNames=(offline-romhr)
case $subTestNum in
  1)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.1 -cfl 0.05 -offline -writesol -romsvds -no-romgs -no-romoffset -no-romsns -romst gnat_lspg
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.1 -cfl 0.05 -online -romhrprep -rdimx 2 -rdimv 5 -rdime 4 -rdimfv 7 -rdimfe 6 -ntsamv 1 -ntsame 1 -sfacv 2 -sface 2 -soldiff -no-romgs -no-romoffset -no-romsns -romst gnat_lspg
    $LAGHOS_SERIAL -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.1 -cfl 0.05 -online -romhr -rdimx 2 -rdimv 5 -rdime 4 -rdimfv 7 -rdimfe 6 -ntsamv 1 -ntsame 1 -sfacv 2 -sface 2 -soldiff -no-romgs -no-romoffset -no-romsns -romst gnat_lspg
    ;;
esac
