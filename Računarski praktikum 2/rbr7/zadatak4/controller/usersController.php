<?php 

require_once __DIR__ . '/../model/libraryservice.class.php';

class UsersController
{
	public function index() 
	{
		$ls = new LibraryService();

		$title = 'Popis svih korisnika knjižnice';
		$userList = $ls->getAllUsers();

		require_once __DIR__ . '/../view/users_index.php';
	}
}; 

?>
