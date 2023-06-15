<?php

// Stvaramo tablice u bazi (ako već ne postoje od ranije).
require_once __DIR__ . '/db.class.php';

create_table_users();
create_table_books();
create_table_loans();

// ------------------------------------------
function create_table_users()
{
	$db = DB::getConnection();

	// Stvaramo tablicu users.
	// Svaki user ima svoj id (automatski će se povećati za svakog novoubačenog korisnika), ime, prezime i password hash.
	try
	{
		$st = $db->prepare(
			'CREATE TABLE IF NOT EXISTS users (' .
			'id int NOT NULL PRIMARY KEY AUTO_INCREMENT,' .
			'name varchar(50) NOT NULL,' .
			'surname varchar(50) NOT NULL,' .
			'password varchar(255) NOT NULL)'
		);

		$st->execute();
	}
	catch( PDOException $e ) { exit( "PDO error (create_table_users): " . $e->getMessage() ); }

	echo "Napravio tablicu users.<br />";
}


function create_table_books()
{
	$db = DB::getConnection();

	// Stvaramo tablicu books.
	// Svaka knjiga ima svoj id (automatski će se povećati za svaku novoubačenu knjigu), ime autora i naslov.
	try
	{
		$st = $db->prepare(
			'CREATE TABLE IF NOT EXISTS books (' .
			'id int NOT NULL PRIMARY KEY AUTO_INCREMENT,' .
			'author varchar(50) NOT NULL,' .
			'title varchar(50) NOT NULL)'
		);

		$st->execute();
	}
	catch( PDOException $e ) { exit( "PDO error (create_table_books): " . $e->getMessage() ); }

	echo "Napravio tablicu books.<br />";
}


function create_table_loans()
{
	$db = DB::getConnection();

	// Stvaramo tablicu loans.
	// Svaka posudba ima svoj id (automatski će se povećati za svaku novoubačenu knjigu), id korisnika koji je posudio
	// knjigu, id knjige koja se posuđuje, te datum isteka posudbe.
	try
	{
		$st = $db->prepare(
			'CREATE TABLE IF NOT EXISTS loans (' .
			'id int NOT NULL PRIMARY KEY AUTO_INCREMENT,' .
			'id_user INT NOT NULL,' .
			'id_book INT NOT NULL,' .
			'lease_end DATE NOT NULL)'
		);

		$st->execute();
	}
	catch( PDOException $e ) { exit( "PDO error (create_table_loans): " . $e->getMessage() ); }

	echo "Napravio tablicu loans.<br />";
}

?>
