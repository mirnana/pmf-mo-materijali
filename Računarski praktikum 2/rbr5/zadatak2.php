<?php 
// .. Funkcije za provjeru ispravnosti boje u stringu
function isValidColorHex( $boja )
{
	return preg_match( '/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/', $boja );	
}

function isValidColorNamed( $boja )
{
	return preg_match( '/^[a-zA-Z]{1,20}$/', $boja );
}

// .. Defaultna boja pozadine
$boja = 'white';
$error = false;
$errorMessage = '';
$errorCausedBy = '';

// .. Provjeri je li postavljen cookie, ako je - učitaj boju iz cookie-a.
if( isset( $_COOKIE['boja'] ) )
{
	$boja = $_COOKIE['boja'];
	if( !isValidColorHex( $boja ) && !isValidColorNamed( $boja ) )
	{
		// .. Netko je otrovao kolačić :)
		$error = true;
		$errorMessage = 'Kolačić se pokvario! Resetiram boju na bijelu.';
		$errorCausedBy = $_COOKIE['boja'];
		$boja = 'white';
	}
}

if( isset( $_POST['bojaTextbox'] ) && $_POST['bojaTextbox'] !== '' )
{
	// .. Provjeri da li je POST-om poslana boja postavljena u textboxu
	// .. Boja mora biti u formatu #HHH ili #HHHHHH, gdje su H hex. znamenke. Format provjeravamo reg. izrazom:
	if( !isValidColorHex( $_POST['bojaTextbox'] ) )
	{
		$error = true;
		$errorMessage = 'Boja unesena u textbox nije u ispravnom formatu! Resetiram na bijelu.';
		$errorCausedBy = $_POST['bojaTextbox'];
		$boja = 'white';
	}
	else
		$boja = $_POST['bojaTextbox'];
}
else if( isset( $_POST['bojaSelect'] ) )
{
	// .. Provjeri da li je POST-om poslana boja postavljena u selectu
	if( !isValidColorNamed( $_POST['bojaSelect'] ) )
	{
		$error = true;
		$errorMessage = 'Boja unesena u select nije u ispravnom formatu! Resetiram na bijelu.';
		$errorCausedBy = $_POST['bojaSelect'];
		$boja = 'white';
	}
	else
		$boja = $_POST['bojaSelect'];
}

// .. Spremi boju u COOKIE. Ističe za 60*60*24*30 sekundi, tj. za 30 dana.
setcookie( 'boja', $boja, time()+60*60*24*30 );


?> 
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8" />
	<title>Zadatak 2</title>
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
	<?php 
		if( $error )
		{
			echo '<br /><br />';
			echo '<p style="color: red;">' . htmlentities( $errorMessage, ENT_QUOTES ) . '</p>';
			echo '<p>Grešku je uzrokovala "boja" "' . htmlentities( $errorCausedBy, ENT_QUOTES ) . '".</p>';
		}
	?>
</body>
</html>

