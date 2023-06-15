<?php

class LibraryService
{
	function getUserById( $id )
	{
		try
		{
			$db = DB::getConnection();
			$st = $db->prepare( 'SELECT id, name, surname, password FROM users WHERE id=:id' );
			$st->execute( array( 'id' => $id ) );
		}
		catch( PDOException $e ) { exit( 'PDO error ' . $e->getMessage() ); }

		$row = $st->fetch();
		if( $row === false )
			return null;
		else
			return new User( $row['id'], $row['name'], $row['surname'], $row['password'] );
	}


	function getAllUsers( )
	{
		try
		{
			$db = DB::getConnection();
			$st = $db->prepare( 'SELECT id, name, surname, password FROM users ORDER BY surname' );
			$st->execute();
		}
		catch( PDOException $e ) { exit( 'PDO error ' . $e->getMessage() ); }

		$arr = array();
		while( $row = $st->fetch() )
		{
			$arr[] = new User( $row['id'], $row['name'], $row['surname'], $row['password'] );
		}

		return $arr;
	}


	function getBookById( $id )
	{
		try
		{
			$db = DB::getConnection();
			$st = $db->prepare( 'SELECT id, author, title FROM books WHERE id=:id' );
			$st->execute( array( 'id' => $id ) );
		}
		catch( PDOException $e ) { exit( 'PDO error ' . $e->getMessage() ); }

		$row = $st->fetch();
		if( $row === false )
			return null;
		else
			return new Book( $row['id'], $row['author'], $row['title'] );
	}


	function getBooksByAuthorName( $author )
	{
		try
		{
			$db = DB::getConnection();
			$st = $db->prepare( 'SELECT id, author, title FROM books WHERE author=:author ORDER BY title' );
			$st->execute( array( 'author' => $author ) );
		}
		catch( PDOException $e ) { exit( 'PDO error ' . $e->getMessage() ); }

		$arr = array();
		while( $row = $st->fetch() )
		{
			$arr[] = new Book( $row['id'], $row['author'], $row['title'] );
		}

		return $arr;
	}


	function getAllBooks()
	{
		try
		{
			$db = DB::getConnection();
			$st = $db->prepare( 'SELECT id, author, title FROM books ORDER BY author' );
			$st->execute();
		}
		catch( PDOException $e ) { exit( 'PDO error ' . $e->getMessage() ); }

		$arr = array();
		while( $row = $st->fetch() )
		{
			$arr[] = new Book( $row['id'], $row['author'], $row['title'] );
		}

		return $arr;
	}

	function getLoansByUserId( $id_user )
	{
		try
		{
			$db = DB::getConnection();
			$st = $db->prepare( 'SELECT id, id_user, id_book, lease_end FROM loans WHERE id_user=:id' );
			$st->execute( array( 'id' => $id_user ) );
		}
		catch( PDOException $e ) { exit( 'PDO error ' . $e->getMessage() ); }

		$arr = array();
		while( $row = $st->fetch() )
		{
			$arr[] = new Loan( $row['id'], $row['id_user'], $row['id_book'], $row['lease_end'] );
		}

		return $arr;
	}

	function getAllAvailableBooks()
	{
		// Vraća sve knjige koje već nisu posuđene. Malo napredniji SQL upit :)
		try
		{
			$db = DB::getConnection();
			$st = $db->prepare( 'SELECT id, author, title FROM books WHERE id NOT IN (SELECT id_book FROM loans) ORDER BY author' );
			$st->execute();
		}
		catch( PDOException $e ) { exit( 'PDO error ' . $e->getMessage() ); }

		$arr = array();
		while( $row = $st->fetch() )
		{
			$arr[] = new Book( $row['id'], $row['author'], $row['title'] );
		}

		return $arr;
	}

	function makeNewLoan( $id_user, $id_book, $lease_end )
	{
		// Provjeri prvo jel postoje taj user i ta knjiga
		try
		{
			$db = DB::getConnection();
			$st = $db->prepare( 'SELECT * FROM users WHERE id=:id' );
			$st->execute( array( 'id' => $id_user ) );
		}
		catch( PDOException $e ) { exit( 'PDO error ' . $e->getMessage() ); }

		if( $st->rowCount() !== 1 )
			throw new Exception( 'makeNewLoan :: User with the given id_user does not exist.' );


		try
		{
			$db = DB::getConnection();
			$st = $db->prepare( 'SELECT * FROM books WHERE id=:id' );
			$st->execute( array( 'id' => $id_book ) );
		}
		catch( PDOException $e ) { exit( 'PDO error ' . $e->getMessage() ); }

		if( $st->rowCount() !== 1 )
			throw new Exception( 'makeNewLoan :: Book with the given id_book does not exist.' );


		// Sad napokon možemo stvoriti novi loan (možda bi trebalo provjeriti i da ta knjiga nije već posuđena...)
		try
		{
			$db = DB::getConnection();
			$st = $db->prepare( 'INSERT INTO loans(id_user, id_book, lease_end) VALUES (:id_user, :id_book, :lease_end)' );
			$st->execute( array( 'id_user' => $id_user, 'id_book' => $id_book, 'lease_end' => $lease_end ) );
		}
		catch( PDOException $e ) { exit( 'PDO error ' . $e->getMessage() ); }
	}


	function returnLoan( $id_loan )
	{
		// Vraća posuđenu knjigu tako da obriše odabrani redak iz tablice
		try
		{
			$db = DB::getConnection();
			$st = $db->prepare( 'DELETE FROM loans WHERE id=:id' );
			$st->execute( array( 'id' => $id_loan ) );
		}
		catch( PDOException $e ) { exit( 'PDO error ' . $e->getMessage() ); }	
	}
};

?>

