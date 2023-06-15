<?php 
/*
	Ovo je rješenje koje ne koristi validaciju korisničkih podataka.
	Ne postavljati na internet.
*/

// .. Defaultna boja pozadine
$boja = 'white';

// .. Provjeri je li postavljen cookie, ako je - učitaj boju iz cookie-a.
if( isset( $_COOKIE['boja'] ) )
	$boja = $_COOKIE['boja'];

if( isset( $_POST['bojaTextbox'] ) && $_POST['bojaTextbox'] !== '' )
{
	// .. Provjeri da li je POST-om poslana boja postavljena u textboxu
	$boja = $_POST['bojaTextbox'];
}
else if( isset( $_POST['bojaSelect'] ) )
{
	// .. Provjeri da li je POST-om poslana boja postavljena u selectu
	$boja = $_POST['bojaSelect'];
}

// .. Spremi boju u COOKIE. Ističe za 60*60*24*30 sekundi, tj. za 30 dana.
setcookie( 'boja', $boja, time()+60*60*24*30 );


?> 
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8" />
	<title>Zadatak 1</title>
	<style>body { background-color: <?php echo $boja;?>; }</style>
</head>
<body>
	<form action="<?php echo $_SERVER['PHP_SELF'];?>" method="post">
		<label for="bojaTextbox">Unesi HTML kod boje pozadine (počinje sa #):</label>		
		<input type="text" name="bojaTextbox" id="bojaTextbox" value="" />

		<br />

		<label for="bojaSelect">Odaberi neku boju iz padajućeg izbornika:</label>
		<select name="bojaSelect" id="bojaSelect">
			<option value="blue" selected>Plava</option>
			<option value="green">Zelena</option>
			<option value="yellow">Žuta</option>
			<option value="white">Bijela</option>
		</select>
		
		<br />

		<button type="reset">Resetiraj!</button>
		<button type="submit">Promijeni!</button>
	</form>
</body>
</html>