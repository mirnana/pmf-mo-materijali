<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8" />
	<title>Zadatak 2</title>
</head>
<body>
	<?php
		$n = 10; $str_len = 5;

		// .. Slucajno generiraj n stringova
		for( $i = 0; $i < $n; ++$i )
		{
			// .. Generiraj i-ti string
			$str = '';
			for( $j = 0; $j < $str_len; ++$j )
				$str .= chr( rand(0, 25) + ord('a') ); // Zalijepi slučajno odabrano malo slovo; ord('a')=97=ascii vrijednost od a. Još bolje: chr( rand( ord('a'), ord('z') ) ).

			$polje[$i] = $str;
		}
	?>

	<p>
		Niz prije sortiranja: 
		<?php
			for( $i = 0; $i < $n; ++$i )
				echo $polje[$i] . " ";
		?>
	</p>

	<?php
		// Sortiranje
		for( $i = 0; $i < $n; ++$i )
			for( $j = $i+1; $j < $n; ++$j )
				if( strcmp( $polje[$i], $polje[$j] ) > 0 )
				{
					$temp = $polje[$i];
					$polje[$i] = $polje[$j];
					$polje[$j] = $temp;
				}
	?>	

	<p>
		Niz nakon sortiranja: 
		<?php
			for( $i = 0; $i < $n; ++$i )
				echo $polje[$i] . ' ';
		?>
	</p>

</body>
</html> 
