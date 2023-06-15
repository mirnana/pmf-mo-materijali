<?php

require_once __DIR__ . '/../app/database/db.class.php';
require_once __DIR__ . '/user.class.php';

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
};

?>

