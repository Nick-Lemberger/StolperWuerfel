<?php

$sock = socket_create(AF_INET, SOCK_DGRAM, SOL_UDP);

/*while (true)
{*/
    $xt = 100;
    $yt = 100;

    `scrot -q 1 -o -a $xt,$yt,64,32 /tmp/test.png`;
    $im = imagecreatefrompng('/tmp/test.png');
    $im = imagerotate($im, 180, 0);
    send_image_to_port($sock, $im, 2000);
    imagedestroy($im);
//}


function send_image_to_port($sock, $im, $port)
{
    $panel_width_bits = 6;
    $panel_height_bits = 5;

    $width = imagesx($im);
    $height = imagesy($im);

    for ($y = 0; $y < $height; $y++)
    {
        $msg = "";
        // header
        $msg .= pack("CC", 4, $y);

        for ($x = 0; $x < $width; $x++)
        {
            $color = imagecolorat($im, $x, $y);
            $color_tran = imagecolorsforindex($im, $color);

            $red = $color_tran['red'];
            $green = $color_tran['green'];
            $blue = $color_tran['blue'];

            $r = ($red  * 1 >> 3) & 0x1f;
            $g = (($green  * 1 >> 2) & 0x3f) << 5;
            $b = (($blue  * 1 >> 3) & 0x1f) << 11;

            // data
            $msg .= pack("n", ($r | $g | $b));
        }

        $len = strlen($msg);
        socket_sendto($sock, $msg, $len, 0, '192.168.178.50', $port);
        usleep(500000);
    }
}


socket_close($sock);



