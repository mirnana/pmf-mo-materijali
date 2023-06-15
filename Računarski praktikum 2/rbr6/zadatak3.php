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

// Dohvati sve podatke o studentima iz tablice.
$st = $db->query( 'SELECT JMBAG,Ime,Prezime,Ocjena FROM Studenti' );
?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="utf8">
	<title>Zadatak 3</title>
	<style>table, td, th { border: 1px solid black; }</style>
</head>
<body>
	<table>
		<tr><th>JMBAG</th><th>Ime</th><th>Prezime</th><th>Ocjena</th></tr>

		<?php 
			// Protrči po svim dohvaćenim studentima iz tablice.
			foreach( $st->fetchAll() as $row )
				echo '<tr>' .
				     '<td>' . $row[ 'JMBAG' ] . '</td>' .
				     '<td>' . $row[ 'Ime' ] . '</td>' .
				     '<td>' . $row[ 'Prezime' ] . '</td>' .
				     '<td>' . $row[ 'Ocjena' ] . '</td>' .
				     '</tr>';
		?>
	</table>
</body>
</html>
 
