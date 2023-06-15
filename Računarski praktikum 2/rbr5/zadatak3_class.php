<?php 
/*
	Rješenje Zadatka 3 pomoću objektno-orijentiranog programiranja.
	Cijelo rješenje je stavljeno u jednu datoteku.
	Proučite prvo "main" na dnu programa, pa onda funkciju run.
*/ 

class PogodiBroj
{
	protected $zamisljeniBroj, $imeIgraca, $brojPokusaja, $gameOver;
	protected $errorMsg;

	// Imena konstanti dolaze BEZ $
	const ZAMISLJENI_JE_VECI = -1, ZAMISLJENI_JE_MANJI = 1, ZAMISLJENI_JE_ISTI = 0;

	function __construct()
	{
		// Generiramo slučajan broj kojeg treba pogoditi
		$this->imeIgraca = false;
		$this->zamisljeniBroj = rand( 1, 100 );
		$this->brojPokusaja = 0;
		$this->gameOver = false;
		$this->errorMsg = false;
	}

	function ispisiFormuZaIme()
	{
		// Ispisi formu koja ucitava ime igraca
		?>

		<!DOCTYPE html>
		<html>
		<head>
			<meta charset="utf-8">
			<title>Pogađanje brojeva - Dobro došli!</title>
		</head>
		<body>
			<form method="post" action="<?php echo htmlentities( $_SERVER['PHP_SELF']); ?>">
				Unesite svoje ime: <input type="text" name="imeIgraca" />
				<button type="submit">Pošalji!</button>
			</form>

			<?php if( $this->errorMsg !== false ) echo '<p>Greška: ' . htmlentities( $this->errorMsg ) . '</p>'; ?>
		</body>
		</html>

		<?php
	}


	function ispisiFormuZaPogadjanjeBroja( $prethodniPokusaj )
	{
		// Ispisuje formu za pogađanje broja + poruku o prethodnom pokušaju.
		// Povećaj brojač pokušaja -- brojim sad i neuspješne pokušaje.
		++$this->brojPokusaja;

		?>
		<!DOCTYPE html>
		<html>
		<head>
			<meta charset="utf-8">
			<title>Pogađanje brojeva - Probaj pogoditi!</title>
		</head>
		<body>
			<p>
				Dobro došao, <?php echo htmlentities( $this->imeIgraca ); ?>!
				<br />
				<?php if( $prethodniPokusaj === PogodiBroj::ZAMISLJENI_JE_VECI  ) echo 'Moj broj je veći!<br />' ?>
				<?php if( $prethodniPokusaj === PogodiBroj::ZAMISLJENI_JE_MANJI ) echo 'Moj broj je manji!<br />' ?>
			</p>

			<form method="post" action="<?php echo htmlentities( $_SERVER['PHP_SELF']); ?>">
				Koji broj između 1 i 100 sam zamislio? 
				<br />
				Pokušaj #<?php echo $this->brojPokusaja; ?>:
				<input type="text" name="pokusaj" />
				<button type="submit">Pogodi!</button>
			</form>

			<?php if( $this->errorMsg !== false ) echo '<p>Greška: ' . htmlentities( $this->errorMsg ) . '</p>'; ?>
		</body>
		</html>

		<?php
	}


	function ispisiCestitku()
	{
		?>
		<!DOCTYPE html>
		<html>
		<head>
			<meta charset="utf-8">
			<title>Pogađanje brojeva - Bravo!</title>
		</head>
		<body>
			<p>
				Bravo, <?php echo htmlentities( $this->imeIgraca ); ?>!
				<br />
				Pogodio si moj zamišljeni broj <?php echo $this->zamisljeniBroj; ?> 
				u samo <?php echo $this->brojPokusaja; ?> pokušaja!
			</p>
		</body>
		</html>

		<?php
	}


	function get_imeIgraca()
	{
		// Je li već definirano ime igrača?
		if( $this->imeIgraca !== false )
			return $this->imeIgraca;

		// Možda nam se upravo sad šalje ime igrača?
		if( isset( $_POST['imeIgraca'] ) )
		{
			// Šalje nam se ime igrača. Provjeri da li se sastoji samo od slova.
			if( !preg_match( '/^[a-zA-Z]{1,20}$/', $_POST['imeIgraca'] ) )
			{
				// Nije dobro ime. Dakle nemamo ime igrača.
				$this->errorMsg = 'Ime igrača treba imati između 1 i 20 slova.';
				return false;
			}
			else
			{
				// Dobro je ime. Spremi ga u objekt.
				$this->imeIgraca = $_POST['imeIgraca'];
				return $this->imeIgraca;
			}
		}

		// Ne šalje nam se sad ime. Dakle nemamo ga uopće.
		return false;
	}


	function obradiPokusaj()
	{
		// Vraća false ako nije bio pokušaj pogađanja, ili je bio neispravan pokušaj pogađanja.
		// Inače, vraća 0 ako su brojevi isti, 1 ako je pokušaj > zamišljeni broj, -1 ako je pokušaj < zamišljeni broj.

		// Da li je igrač uopće pokusao pogađati broj?
		if( isset( $_POST['pokusaj'] ) )
		{
			// Je. Da li je pokušaj broj između 1 i 100?
			$options = array( 'options' => array( 'min_range' => 1, 'max_range' => 100 ) );

			if( filter_var( $_POST['pokusaj'], FILTER_VALIDATE_INT, $options ) === false )
			{
				// Nije unesen broj između 1 i 100.
				$this->errorMsg = 'Trebate unijeti broj između 1 i 100.';
				return false;
			}
			else
				$pokusaj = (int) $_POST['pokusaj'];

			// Ispravan je pokušaj. Je li veći/manji/isti kao zamišljeni broj?
			if( $pokusaj === $this->zamisljeniBroj )
				return PogodiBroj::ZAMISLJENI_JE_ISTI;
			else if( $pokusaj < $this->zamisljeniBroj )
				return PogodiBroj::ZAMISLJENI_JE_VECI;
			else
				return PogodiBroj::ZAMISLJENI_JE_MANJI;
		}

		// Igrač nije pokušao pogoditi broj.
		return false;
	}


	function isGameOver() { return $this->gameOver; }


	function run()
	{
		// Funkcija obavlja "jedan potez" u igri.
		// Prvo, resetiraj poruke o greški.
		$this->errorMsg = false;

		// Prvo provjeri jel imamo uopće ime igraca
		if( $this->get_imeIgraca() === false )
		{
			// Ako nemamo ime igrača, ispiši formu za unos imena i to je kraj.
			$this->ispisiFormuZaIme();
			return;
		}

		// Dakle imamo ime igrača.
		// Ako je igrač pokušao pogoditi broj, provjerimo što se dogodilo s tim pokušajem.
		$rez = $this->obradiPokusaj();

		if( $rez === PogodiBroj::ZAMISLJENI_JE_ISTI )
		{
			// Ako je igrač pogodio, ispiši mu čestitku.
			$this->ispisiCestitku();
			$this->gameOver = true;
		}
		else
			$this->ispisiFormuZaPogadjanjeBroja( $rez );
	}
};


// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// Sad ide "glavni program" -- skroz generički, isti za svaku moguću igru.

// U $_SESSION ćemo čuvati cijeli objekt tipa PogodiBroj.
// U tom slučaju definicija klase treba biti PRIJE session_start();
session_start();

if( !isset( $_SESSION['igra'] ) )
{
	// Ako igra još nije započela, stvori novi objekt tipa PogodiBroj i spremi ga u $_SESSION
	$igra = new PogodiBroj();
	$_SESSION['igra'] = $igra;
}
else
{
	// Ako je igra već ranije započela, dohvati ju iz $_SESSION-a	
	$igra = $_SESSION['igra'];
}

// Izvedi jedan korak u igri, u kojoj god fazi ona bila.
$igra->run();

if( $igra->isGameOver() )
{
	// Kraj igre -> prekini session.
	session_unset();
	session_destroy();
}
else
{
	// Igra još nije gotova -> spremi trenutno stanje u SESSION
	$_SESSION['igra'] = $igra;	
}
