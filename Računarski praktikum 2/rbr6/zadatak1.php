<?php 
$fileName = 'users.txt';
// fileName vjerojatno želimo smjestiti izvan public_html, tako da ne bude dostupan preko weba.
// Na primjer: 
//    ako je skripta zadatak1.php smještena u public_html/rp2/zadatak1.php,
//    onda __DIR__ = /student1/username/public_html/rp2
//    pa možemo staviti $fileName = __DIR__ . '/../../aux/users.txt'
//    što daje $fileName = /student1/username/aux/users.txt.

// Procitaj popis svih usera iz datoteke 'users.txt'
if( !file_exists($fileName) || ($fileContent = file_get_contents( $fileName )) === false )
{
	// Ne mogu pročitati file (možda još ni ne postoji?). Napravi prazan popis.
	$popis = array();
}
else
{
	// U stringu fileContent je popis dosadašnjih usera, odvojena su zarezima.
	// Napravi of njih polje.
	$popis = explode( ',', $fileContent ); 
}

// Da li se šalje novo ime? Da li je u ispravnom formatu?
if( isset( $_POST['ime'] ) && preg_match( '/^[a-zA-Z]{1,20}$/', $_POST['ime'] ) )
{
	// OK je ime. Dodaj ga u popis.
	$len = count( $popis );
	$popis[$len] = $_POST['ime'];

	// Ako sad ima više od 5 imena, obriši nulto
	if( $len >= 5 )
		unset( $popis[0] );

	// Spremi novi popis u file.
	// Prvo pretvori popis u string, imena su odvojena zarezima.
	$str = implode( ',', $popis );

	if( file_put_contents( $fileName, $str ) === false )
		exit( 'Ne mogu pisati u datoteku users.txt' );
}

?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Zadatak 1</title>
</head>
<body>
	<form method="post" action="<?php echo htmlentities( $_SERVER['PHP_SELF']); ?>">
		Unesi svoje ime:
		<input type="text" name="ime" />
		<button type="submit">Pošalji</button>
	</form>

	<p>
		Popis zadnjih max. 5 korisnika:
		<ul>
			<?php 
				foreach( $popis as $ime )
					echo '<li>' . htmlentities( $ime ) . '</li>';
			?>
		</ul>
	</p>

</body>
</html>
