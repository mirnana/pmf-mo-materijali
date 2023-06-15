<?php 

// Spoji se na bazu
$user = ''; // unesite username sa papira
$pass = ''; // unesite password sa papira

try 
{
	// zamijenite HOST imenom servera za rp2 ili njegovom ip adresom
	// zamijenite PREZIME svojim prezimenom (malim slovima), tj. imenom vaše baze na rp2 serveru.
	$db = new PDO( 'mysql:host=HOST;dbname=PREZIME;charset=utf8', $user, $pass );
} 
catch( PDOException $e ) 
{
	echo "Greška: " . $e->getMessage(); exit();
}

// Nema greške, ali ni ocjena još nije postavljena.
$error = false; $ocjena = false;

// Da li je u POST zahtjevu stigla ocjena iz forme?
if( isset( $_POST['ocjena'] ) )
{
	// Provjeri da li je zaista ocjena broj između 1 i 5
	if( !preg_match( '/^[1-5]$/', $_POST['ocjena'] ) )
		$error = true; // Ocjena je krivo unesena
	else
		$ocjena = (int) $_POST['ocjena']; // Ocjena je ispravno unesena
}

// Pripremi SQL i izvrši upit u bazu.
if( !$error && $ocjena !== false )
{
	$st = $db->prepare( 'SELECT JMBAG,Ime,Prezime,Ocjena FROM Studenti WHERE Ocjena LIKE :ocjena' );
	$st->execute( array( 'ocjena' => $ocjena ) );
}
?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="utf8">
	<title>Zadatak 4</title>
	<style>table, td, th { border: 1px solid black; }</style>
</head>
<body>
	<?php 
		if( !$error && $ocjena !== false )
		{
			// Ako nema greške i ocjena je ispravna, iscrtaj tablicu.
			?>
			<table>
				<tr><th>JMBAG</th><th>Ime</th><th>Prezime</th><th>Ocjena</th></tr>

				<?php 
					if( !$error && $ocjena !== false )
					{
						// Protrči po svim studentima koji imaju tu ocjenu.
						foreach( $st->fetchAll() as $row )
							echo '<tr>' .
							     '<td>' . $row[ 'JMBAG' ] . '</td>' .
							     '<td>' . $row[ 'Ime' ] . '</td>' .
							     '<td>' . $row[ 'Prezime' ] . '</td>' .
							     '<td>' . $row[ 'Ocjena' ] . '</td>' .
							     '</tr>';
					}
				?>
			</table>
			<?php 
		}
		else if( $error )
		{
			// Ako ima greške, znači da je ocjena bila krivo unesena.
			echo '<p>Ocjena je broj između 1 i 5.</p>';
		}

	// U svakom slučaju nacrtaj formu koja će omogućiti pretraživanje po ocjenama.
	?>

	<form method="post" action="<?php echo htmlentities( $_SERVER['PHP_SELF'] ); ?>">
		Koja ocjena vas zanima?
		<input type="text" name="ocjena" />
		<button type="submit">Pošalji!</button>
	</form>
</body>
</html>
