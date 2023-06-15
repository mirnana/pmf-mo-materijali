<?php 
session_start();

if( isset( $_POST['ime'] ) )
{
	// Prosljeđuje nam se ime korisnika od strane zadatak3_index.php
	// Recimo da smije imati između 3 i 20 slova.
	if( preg_match( '/^[a-zA-Z]{3,20}$/', $_POST['ime'] ) )
	{
		// Zapamti korisnikovo ime u sesiji.
		$_SESSION['ime'] = $_POST['ime'];
	}
	else
	{
		// Korisnikovo ime nije validno. Preusmjeri ga na index.
		header( 'Location: zadatak3_index.php' );
		exit;
	}
}

if( !isset( $_SESSION['ime'] ) || !isset( $_SESSION['broj'] ) )
{
	// Netko je direktno probao pristupiti zadatak3_pogodi.php
	// bez da je prvo unio ime preko zadatak3_index.php
	// Preusmjeri ga na index.
	header( 'Location: zadatak3_index.php' );
	exit;
}

// Dakle sad su definirani i $_SESSION['ime'] i $_SESSION['broj'].
// Izvadi ih u varijable da bude lakše s njima raditi.
$ime  = $_SESSION['ime'];
$zamisljeniBroj = (int) $_SESSION['broj'];
$brojPokusaja = $_SESSION['brojPokusaja'];

$error = false;
$errorMessage = '';
unset( $pokusaj );
if( isset( $_POST['pokusaj'] ) )
{
	// Provjeri jel korisnik unio broj izmedju 1 i 100 -- DZ: preg_match
	$options = array( 'options' => array( 'min_range' => 1, 'max_range' => 100 ) );

	if( filter_var( $_POST['pokusaj'], FILTER_VALIDATE_INT, $options) === FALSE ) 
	{
		$error = true;
		$errorMessage = 'Trebaš unijeti broj između 1 i 100.';
	}
    else
    {
    	// Dakle unesen je broj izmedju 1 i 100
    	$pokusaj = (int) $_POST['pokusaj'];
    	++$brojPokusaja;
    	$_SESSION['brojPokusaja'] = $brojPokusaja;
    }
}

$pogodio = false;
$pogodioMessage = '';
if( isset( $pokusaj ) )
{
	if( $pokusaj === $zamisljeniBroj )
		$pogodio = true;
	elseif( $pokusaj < $zamisljeniBroj )
		$pogodioMessage = 'Moj zamišljeni broj je veći od ' . $pokusaj . '!';
	else
		$pogodioMessage = 'Moj zamišljeni broj je manji od ' . $pokusaj . '!';
}

?>


<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Zadatak 3 - pogodi</title>
</head>
<body>
	<p>
		Dobro došao, <?php echo htmlentities( $ime, ENT_QUOTES ); ?>!
	</p>

	<p>
		<?php 
			// Provjeri što je bilo s pokušajem pogađanja
			if( $error )
				echo htmlentities( $errorMessage, ENT_QUOTES );
			else if( $pogodio )
			{
				echo 'Bravo! Pogodio si iz ' . $brojPokusaja . ' pokusaja!';
				
				// Završavamo sesiju.
				session_unset();
				session_destroy();
			}
			else
				echo htmlentities( $pogodioMessage, ENT_QUOTES );
		?>
	</p>

	<br />
	<?php 
		if( !$pogodio )
		{ 
			?>
			<form method="post" action="zadatak3_pogodi.php">
				<label for="pokusaj">
					Pokušaj pogoditi broj između 1 i 100 kojeg sam zamislio:
				</label>
				<input type="text" id="pokusaj" name="pokusaj" />
				<button type="submit">Pogodi!</button>
			</form>
			<?php 
		}
	?>
</body>
</html> 
