<?php

// Popunjavamo tablice u bazi "probnim" podacima.
require_once __DIR__ . '/db.class.php';

seed_table_users();
seed_table_books();
seed_table_loans();

// ------------------------------------------
function seed_table_users()
{
	$db = DB::getConnection();

	// Ubaci neke korisnike u tablicu users.
	// Uočimo da ne treba specificirati id koji se automatski poveća kod svakog ubacivanja.
	try
	{
		$st = $db->prepare( 'INSERT INTO users(name, surname, password) VALUES (:name, :surname, :password)' );

		$st->execute( array( 'name' => 'Pero', 'surname' => 'Perić', 'password' => password_hash( 'perinasifra', PASSWORD_DEFAULT ) ) );
		$st->execute( array( 'name' => 'Mirko', 'surname' => 'Mirić', 'password' => password_hash( 'mirkovasifra', PASSWORD_DEFAULT ) ) );
		$st->execute( array( 'name' => 'Slavko', 'surname' => 'Slavić', 'password' => password_hash( 'slavkovasifra', PASSWORD_DEFAULT ) ) );
		$st->execute( array( 'name' => 'Ana', 'surname' => 'Anić', 'password' => password_hash( 'aninasifra', PASSWORD_DEFAULT ) ) );
		$st->execute( array( 'name' => 'Maja', 'surname' => 'Majić', 'password' => password_hash( 'majinasifra', PASSWORD_DEFAULT ) ) );
	}
	catch( PDOException $e ) { exit( "PDO error (seed_table_users): " . $e->getMessage() ); }

	echo "Ubacio korisnike u tablicu users.<br />";
}


function seed_table_books()
{
	$db = DB::getConnection();

	// Ubaci neke knjige u tablicu books.
	// Uočimo da ne treba specificirati id koji se automatski poveća kod svakog ubacivanja.
	try
	{
		$st = $db->prepare( 'INSERT INTO books(author, title) VALUES (:author, :title)' );

		$st->execute( array( 'author' => 'Clarke, Arthur C.', 'title' => 'Rendezvous with Rama' ) );
		$st->execute( array( 'author' => 'Clarke, Arthur C.', 'title' => '2001: A Space Odyssey' ) );
		$st->execute( array( 'author' => 'Asimov, Isaac', 'title' => 'Foundation' ) );
		$st->execute( array( 'author' => 'Asimov, Isaac', 'title' => 'The Gods Themselves' ) );
		$st->execute( array( 'author' => 'Card, Orson Scott', 'title' => 'Ender\'s game' ) );
	}
	catch( PDOException $e ) { exit( "PDO error (seed_table_books): " . $e->getMessage() ); }

	echo "Ubacio knjige u tablicu books.<br />";
}


function seed_table_loans()
{
	$db = DB::getConnection();

	// Ubaci neke posudbe u tablicu loans.
	// Uočimo da ne treba specificirati id posudbe koji se automatski poveća kod svakog ubacivanja.
	// Ovo nije baš pametno raditi preko hardcodiranih id-eva usera i knjiga jer
	// 1. pretpostavlja da će id-ovi biti dodijeljivani baš redom 1, 2, 3, ...
	// 2. teško je vidjeti koji korisnik je posudio koju knjigu (lakše je vidjeti ime korisnika i naslov knjige).
	// Alternativno, mogli bismo pronaći id korisnika s imenom Mirko (select u tablici users)
	// i id knjige s naslovom XYZ (select u tablici books) i onda ih spojiti (kao dolje) u tablici loans.
	try
	{
		$st = $db->prepare( 'INSERT INTO loans(id_user, id_book, lease_end) VALUES (:id_user, :id_book, :lease_end)' );

		$st->execute( array( 'id_user' => 1, 'id_book' => 2, 'lease_end' => '2020-04-11') );
		$st->execute( array( 'id_user' => 3, 'id_book' => 1, 'lease_end' => '2020-05-03') );
		$st->execute( array( 'id_user' => 5, 'id_book' => 4, 'lease_end' => '2020-03-21') );
		$st->execute( array( 'id_user' => 3, 'id_book' => 5, 'lease_end' => '2020-05-12') );
	}
	catch( PDOException $e ) { exit( "PDO error (seed_table_loans): " . $e->getMessage() ); }

	echo "Ubacio knjige u tablicu books.<br />";
}

?>
