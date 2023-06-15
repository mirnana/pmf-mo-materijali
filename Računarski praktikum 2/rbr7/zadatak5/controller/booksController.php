<?php 

require_once __DIR__ . '/../model/libraryservice.class.php';

class BooksController
{
	public function index() 
	{
		$ls = new LibraryService();

		$title = 'Popis svih knjiga knjižnice';
		$bookList = $ls->getAllBooks();

		require_once __DIR__ . '/../view/books_index.php';
	}


	public function search() 
	{
		$title = 'Pretraži knjige po autoru';

		require_once __DIR__ . '/../view/books_search.php';
	}

	public function searchResults() 
	{
		$ls = new LibraryService();


		if( !isset( $_POST['author'] ) || !preg_match( '/^[a-zA-Z ,-.]+$/', $_POST['author'] ) )
		{
			header( 'Location: index.php?rt=books/search');
			exit();
		}

		$title = 'Popis svih knjiga autora ' . $_POST['author'];
		$bookList = $ls->getBooksByAuthorName( $_POST['author'] );

		require_once __DIR__ . '/../view/books_index.php';
	}
}; 

?>
