#!/bin/sh

echo username?
read USERNAME
echo password?
read PASSWORD

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231022170901/20231022170901_mask.png ./train_scrolls/20231022170901/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231022170901/layers/ ./train_scrolls/20231022170901/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231210121321/20231210121321_mask.png ./train_scrolls/20231210121321/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231210121321/layers/ ./train_scrolls/20231210121321/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231221180251/20231221180251_mask.png ./train_scrolls/20231221180251/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231221180251/layers/ ./train_scrolls/20231221180251/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230530164535/20230530164535_mask.png ./train_scrolls/20230530164535/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230530164535/layers/ ./train_scrolls/20230530164535/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231007101615_superseded/20231007101615_mask.png ./train_scrolls/20231007101615/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231007101615_superseded/layers/ ./train_scrolls/20231007101615/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230530172803/20230530172803_mask.png ./train_scrolls/20230530172803/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230530172803/layers/ ./train_scrolls/20230530172803/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230522215721/20230522215721_mask.png ./train_scrolls/20230522215721/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230522215721/layers/ ./train_scrolls/20230522215721/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230620230617/20230620230617_mask.png ./train_scrolls/20230620230617/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230620230617/layers/ ./train_scrolls/20230620230617/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230902141231/20230902141231_mask.png ./train_scrolls/20230902141231/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230902141231/layers/ ./train_scrolls/20230902141231/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231016151000_superseded/20231016151000_mask.png ./train_scrolls/20231016151000/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231016151000_superseded/layers/ ./train_scrolls/20231016151000/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230520175435/20230520175435_mask.png ./train_scrolls/20230520175435/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230520175435/layers/ ./train_scrolls/20230520175435/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230531121653/20230531121653_mask.png ./train_scrolls/20230531121653/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230531121653/layers/ ./train_scrolls/20230531121653/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/stephen-parsons-uploads/verso/Scroll1_part_1_wrap_verso_mask.png ./train_scrolls/verso/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/stephen-parsons-uploads/verso/Scroll1_part_1_wrap_verso_surface_volume/ ./train_scrolls/verso/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
( 
  cd train_scrolls/verso;
  ln -s Scroll1_part_1_wrap_verso_mask.png verso_mask.png
)
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230826170124/20230826170124_mask.png ./train_scrolls/20230826170124/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230826170124/layers/ ./train_scrolls/20230826170124/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230901184804/20230901184804_mask.png ./train_scrolls/20230901184804/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230901184804/layers/ ./train_scrolls/20230901184804/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230813_real_1/20230813_real_1_mask.png ./train_scrolls/20230813_real_1/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230813_real_1/layers/ ./train_scrolls/20230813_real_1/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230901234823/20230901234823_mask.png ./train_scrolls/20230901234823/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230901234823/layers/ ./train_scrolls/20230901234823/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/stephen-parsons-uploads/recto/Scroll1_part_1_wrap_recto_mask.png ./train_scrolls/recto/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/stephen-parsons-uploads/recto/Scroll1_part_1_wrap_recto_surface_volume/ ./train_scrolls/recto/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
( 
  cd train_scrolls/recto;
  ln -s Scroll1_part_1_wrap_recto_mask.png recto_mask.png
)
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230531211425/20230531211425_mask.png ./train_scrolls/20230531211425/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230531211425/layers/ ./train_scrolls/20230531211425/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230531193658/20230531193658_mask.png ./train_scrolls/20230531193658/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230531193658/layers/ ./train_scrolls/20230531193658/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230919113918/20230919113918_mask.png ./train_scrolls/20230919113918/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230919113918/layers/ ./train_scrolls/20230919113918/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230904020426/20230904020426_mask.png ./train_scrolls/20230904020426/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230904020426/layers/ ./train_scrolls/20230904020426/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230601193301/20230601193301_mask.png ./train_scrolls/20230601193301/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230601193301/layers/ ./train_scrolls/20230601193301/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231012184420/20231012184420_mask.png ./train_scrolls/20231012184420/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231012184420/layers/ ./train_scrolls/20231012184420/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230903193206/20230903193206_mask.png ./train_scrolls/20230903193206/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230903193206/layers/ ./train_scrolls/20230903193206/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230929220924_superseded/20230929220924_mask.png ./train_scrolls/20230929220924/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230929220924_superseded/layers/ ./train_scrolls/20230929220924/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230820203112/20230820203112_mask.png ./train_scrolls/20230820203112/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230820203112/layers/ ./train_scrolls/20230820203112/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231005123333_superseded/20231005123333_mask.png ./train_scrolls/20231005123333/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231005123333_superseded/layers/ ./train_scrolls/20231005123333/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230625171244/20230625171244_mask.png ./train_scrolls/20230625171244/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230625171244/layers/ ./train_scrolls/20230625171244/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230620230619/20230620230619_mask.png ./train_scrolls/20230620230619/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230620230619/layers/ ./train_scrolls/20230620230619/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230530212931/20230530212931_mask.png ./train_scrolls/20230530212931/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230530212931/layers/ ./train_scrolls/20230530212931/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230601204340/20230601204340_mask.png ./train_scrolls/20230601204340/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230601204340/layers/ ./train_scrolls/20230601204340/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230813_frag_2/20230813_frag_2_mask.png ./train_scrolls/20230813_frag_2/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230813_frag_2/layers/ ./train_scrolls/20230813_frag_2/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231012173610/20231012173610_mask.png ./train_scrolls/20231012173610/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231012173610/layers/ ./train_scrolls/20231012173610/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230702185753/20230702185753_mask.png ./train_scrolls/20230702185753/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230702185753/layers/ ./train_scrolls/20230702185753/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231106155350_superseded/20231106155350_mask.png ./train_scrolls/20231106155350/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231106155350_superseded/layers/ ./train_scrolls/20231106155350/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231031143850_superseded/20231031143850_mask.png ./train_scrolls/20231031143850/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231031143850_superseded/layers/ ./train_scrolls/20231031143850/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231012184421_superseded/20231012184421_mask.png ./train_scrolls/20231012184421/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231012184421_superseded/layers/ ./train_scrolls/20231012184421/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231012184423_superseded/20231012184423_mask.png ./train_scrolls/20231012184423/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231012184423_superseded/layers/ ./train_scrolls/20231012184423/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231106155351/20231106155351_mask.png ./train_scrolls/20231106155351/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231106155351/layers/ ./train_scrolls/20231106155351/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231031143852/20231031143852_mask.png ./train_scrolls/20231031143852/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231031143852/layers/ ./train_scrolls/20231031143852/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231005123336/20231005123336_mask.png ./train_scrolls/20231005123336/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231005123336/layers/ ./train_scrolls/20231005123336/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231210121320_superseded/20231210121320_mask.png ./train_scrolls/20231210121320/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231210121320_superseded/layers/ ./train_scrolls/20231210121320/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231210121321/20231210121321_mask.png ./train_scrolls/20231210121321/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231210121321/layers/ ./train_scrolls/20231210121321/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231022170901/20231022170901_mask.png ./train_scrolls/20231022170901/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20231022170901/layers/ ./train_scrolls/20231022170901/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8


rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230929220926/20230929220926_mask.png ./train_scrolls/20230929220926/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230929220926/layers/ ./train_scrolls/20230929220926/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230702185753/20230702185753_mask.png ./train_scrolls/20230702185753/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230702185753/layers/ ./train_scrolls/20230702185753/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
