<?php 
/*
	Ovaj kod ilustrira ne-lijepu implementaciju rješenja Zadatka 3.
	DZ: Napišite objektno-orijentirani kod, sličan rješenju Zadatka 4, koji će biti puno elegantniji.
*/

session_start();

// Provjeri je li već zamišljen slučajni broj između 1 i 100
if( !isset( $_SESSION['broj'] ) )
{
	// Slučajno generiraj broj kojeg treba pogoditi i zapamti ga u sessionu.
	$_SESSION['broj'] = rand(1, 100);
	$_SESSION['brojPokusaja'] = 0;
}

if( isset( $_SESSION['ime'] ) )
{
	// Ako je korisnik već ranije unio ime i sad ponovno pristupa
	// ovoj stranici, preusmjerit ćemo ga na zadatak3_pogodi.php
	header('Location: zadatak3_pogodi.php');
	exit;
}

?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Zadatak 3 - index</title>
</head>
<body>
<form action="zadatak3_pogodi.php" method="post">
	<label for="ime">Unesite svoje ime (između 3 i 20 slova):</label>
	<input type="text" id="ime" name="ime" />

	<br />

	<button type="submit">Pošalji</button>
</form> 	
</body>
</html> 
