<?php 

// U db_settings.php su definirani $db_base, $db_user, $db_pass
require_once 'zadatak5_db.php';
require_once 'zadatak5_html.php';


// Funkcija koja procesira što se dogodi nakon klika na gumb "Ulogiraj se!"
function procesiraj_login()
{
	// Provjeri sastoji li se ime samo od slova; ako ne, crtaj login formu.
	if( !isset( $_POST["username"] ) || preg_match( '/[a-zA-Z]{1, 20}/', $_POST["username"] ) )
		crtaj_loginForma();

	// Možda se ne šalje password; u njemu smije biti bilo što.
	if( !isset( $_POST["password"] ) )
		crtaj_loginForma();

	// Sve je OK, provjeri jel ga ima u bazi.
	$db = DB::getConnection();

	try
	{
		$st = $db->prepare( 'SELECT password FROM users WHERE username=:username' );
		$st->execute( array( 'username' => $_POST["username"] ) );
	}
	catch( PDOException $e ) { crtaj_loginForma( 'Greška:' . $e->getMessage() ); return; }

	$row = $st->fetch();

	if( $row === false )
	{
		// Taj user ne postoji, upit u bazu nije vratio ništa.
		crtaj_loginForma( 'Ne postoji korisnik s tim imenom.' );
		return;
	}
	else
	{
		// Postoji user. Dohvati hash njegovog passworda.
		$hash = $row[ 'password'];

		// Da li je password dobar?
		if( password_verify( $_POST['password'], $hash ) )
		{
			// Dobar je. Ulogiraj ga.
			crtaj_uspjesnoUlogiran( $_POST['username' ] );
			return;
		}
		else
		{
			// Nije dobar. Crtaj opet login formu s pripadnom porukom.
			crtaj_loginForma( 'Postoji user, ali password nije dobar.' );
			return;
		}
	}
}


// Funkcija koja procesira što se dogodi nakon klika na gumb "Stvori novog korisnika!"
function procesiraj_novi()
{
	// Provjeri sastoji li se ime samo od slova; ako ne, crtaj login formu.
	if( !isset( $_POST["username"] ) || preg_match( '/[a-zA-Z]{1, 20}/', $_POST["username"] ) )
		crtaj_loginForma();

	// Možda se ne šalje password; u njemu smije biti bilo što.
	if( !isset( $_POST["password"] ) )
		crtaj_loginForma();

	// Sve je OK, provjeri jel ga ima u bazi.
	$db = DB::getConnection();

	try
	{
		$st = $db->prepare( 'SELECT password FROM users WHERE username=:username' );
		$st->execute( array( 'username' => $_POST["username"] ) );
	}	
	catch( PDOException $e ) { crtaj_loginForma( 'Greška:' . $e->getMessage() ); return; }

	if( $st->rowCount() > 0 )
	{
		// Taj korisnik već postoji. Ponovno crtaj login.
		crtaj_loginForma( 'Taj korisnik već postoji.' );
		return;
	}
	else
	{
		// Stvarno nema tog korisnika. Dodaj ga u bazu.
		try
		{
			// Prvo pripremi insert naredbu.
			$st = $db->prepare( 'INSERT INTO users (username, password) VALUES (:username, :hash)' );

			// Napravi hash od passworda kojeg je unio user.
			$hash = password_hash( $_POST["password"], PASSWORD_DEFAULT );

			// Izvrši sad tu insert naredbu. Uočite da u bazu stavljamo hash, a ne $_POST["password"]!
			$st->execute( array( 'username' => $_POST["username"], 'hash' => $hash ) );
		}
		catch( PDOException $e ) { crtaj_loginForma( 'Greška:' . $e->getMessage() ); return; }

		// Sad ponovno nacrtaj login formu, tako da se user proba ulogirati.
		crtaj_loginForma( 'Novi korisnik je uspješno dodan!' );
	}
}


// ---
// --- Glavni dio programa.
// ---


// Provjeri šalje li se username ili password, te da li je tražio novog korisnika ili postojećeg.
if( isset( $_POST["gumb" ] ) && $_POST["gumb"] === "login" )
	procesiraj_login();
else if( isset( $_POST["gumb" ] ) && $_POST["gumb"] === "novi" )
	procesiraj_novi();
else
	crtaj_loginForma();
?> 
