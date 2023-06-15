<?php

class Loan
{
	protected $id, $id_book, $id_user, $lease_end;

	function __construct( $id, $id_user, $id_book, $lease_end )
	{
		$this->id = $id;
		$this->id_user = $id_user;
		$this->id_book = $id_book;
		$this->lease_end = $lease_end;
	}

	function __get( $prop ) { return $this->$prop; }
	function __set( $prop, $value ) { $this->$prop = $value; return $this; }
}

?>

