<?php
interface iUpravljiv
{
	function idiRavno( $x );
	function skreniLijevo();
	function skreniDesno();
};

abstract class Vozilo implements iUpravljiv
{
	protected $smjer = 'N';
	protected $x = 0, $y = 0;
	protected $ime;

	// .. Pomak u x- i y-koordinati za svaki smjer.
	private static $smjer2dx = array( 'N' => 0, 'S' => 0, 'W' => -1, 'E' => 1 );
	private static $smjer2dy = array( 'N' => 1, 'S' => -1, 'W' => 0, 'E' => 0 );

	// .. U koji smjer dođemo kad idemo lijevo od nekog smjera.
	private static $smjerLijevo = array( 'N' => 'W', 'W' => 'S', 'S' => 'E', 'E' => 'N' );
	private static $smjerDesno = array( 'W' => 'N', 'S' => 'W', 'E' => 'S', 'N' => 'E' );

	// .. Konstruktor.
	function __construct( $ime ) { $this->ime = $ime; }

	// .. Funkcije članice.
	function gdjeSam() { echo $this->ime . ' -> (' . $this->x . ', ' . $this->y . ')'; }

	// .. Može se ispisivati i sa echo ako redefiniramo magičnu metodu __toString.
	function __toString() { return $this->ime . ' -> (' . $this->x . ', ' . $this->y . ')'; }

	// .. Funkcije iz interfacea iUpravljiv.
	function idiRavno( $k ) { $this->x += $k*Vozilo::$smjer2dx[$this->smjer]; $this->y += $k*Vozilo::$smjer2dy[$this->smjer]; }
	function skreniLijevo() { $this->smjer = Vozilo::$smjerLijevo[$this->smjer]; }
	function skreniDesno() { $this->smjer = Vozilo::$smjerDesno[$this->smjer]; }
};

class Auto extends Vozilo
{
	private $potrosnja = 0;

	// .. Funkcija koja se nalazi samo u naslijeđenoj klasi.
	public function potrosenBenzin() { return $this->potrosnja; }

	// .. Redefinicija funkcije iz interfacea za nasljeđenu klasu.
	public function idiRavno( $k ) { $this->potrosnja += $k / 10.0; parent::idiRavno($k); }
};

class Tramvaj extends Vozilo
{
	private $linija;

	// .. Specijalni konstruktor za naslijeđenu klasu.
	function __construct( $ime, $l ) { $this->linija = $l; parent::__construct( $ime ); }

	// .. Funkcija koja se nalazi samo u naslijeđenoj klasi.
	function linija() { return $this->linija; }
};


// .. Stvorimo polje sa 10 raznih vozila.
$polje[0] = new Auto( 'yugo' );
$polje[1] = new Tramvaj( 'petica', 5 );
$polje[2] = new Auto( 'bmw' );

for( $i = 3; $i < 7; ++$i )
	$polje[$i] = new Tramvaj( 'tramvaj broj ' . $i, $i );

for( $i = 7; $i < 10; ++$i )
	$polje[$i] = new Auto( 'trabant #' . $i );


// .. Iteriranje po polju. Može i običnom for-petljom.
foreach( $polje as $vozilo )
{
	$vozilo->idiRavno( 10 );
	$vozilo->skreniDesno();
	$vozilo->idiRavno( 10 );
	$vozilo->skreniDesno();
	$vozilo->idiRavno( 20 );
	$vozilo->skreniLijevo();
	$vozilo->skreniLijevo();
	$vozilo->skreniLijevo();
	$vozilo->idiRavno( 5 );
}

?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8" />
	<title>Zadatak 4</title>
</head>
<body>
	<?php
		// Auto (yugo)
		echo '<p>';
		$polje[0]->gdjeSam();
		echo '<br />Potrošen benzin: ' . $polje[0]->potrosenBenzin();
		echo '</p>';
		
		// Tramvaj (petica)
		echo '<p>';
		$polje[1]->gdjeSam();
		echo '<br />Linija: ' . $polje[1]->linija();
		echo '</p>';

		// Auto (bmw)
		echo '<p>';
		echo $polje[2]; // Može i ovako jer postoji __toString().
		echo '<br />Potrošen benzin: ' . $polje[2]->potrosenBenzin();
		echo '</p>';

		// Ostala vozila
		for( $i = 3; $i < 10; ++$i )
		{
			echo '<p>';
			echo $polje[$i];
			
			if( $polje[$i] instanceof Auto ) // Detekcija tipa!
				echo '<br />Potrošen benzin: ' . $polje[$i]->potrosenBenzin();
			else if( $polje[$i] instanceof Tramvaj ) // Detekcija tipa!
				echo '<br />Linija: ' . $polje[$i]->linija();
			
			echo '</p>';
		}
	?>
</body>
</html> 
