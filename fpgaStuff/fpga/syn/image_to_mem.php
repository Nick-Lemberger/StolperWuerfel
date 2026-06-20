<?php
$im = imagecreatefrompng('no_signal.png');

//$im = imagecreatefromjpeg('Bliss.jpg');
$width = imagesx($im);
$height = imagesy($im);

for ($y = 0; $y < $height; $y++)
{
	for ($x = 0; $x < $width; $x++)
	{

		$color = imagecolorat($im, $x, $y);
		$color_tran = imagecolorsforindex($im, $color);

		$red = $color_tran['red'];
		$green = $color_tran['green'];
		$blue = $color_tran['blue'];

		$r = ($red >> 3) & 0x1f;
		$g = (($green >> 2) & 0x3f) << 5;
		$b = (($blue >> 3) & 0x1f) << 11;

		echo dechex($r | $g | $b) . "\n";
   }
}
