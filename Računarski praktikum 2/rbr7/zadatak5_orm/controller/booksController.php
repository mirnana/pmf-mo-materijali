<?php 

require_once __DIR__ . '/../model/book.class.php';

class BooksController
{
    public function index()
    {
        $title = 'Popis svih knjiga';
        $bookList = Book::all();

        require_once __DIR__ . '/../view/books_index.php';
    }


    public function search()
    {
        $title = 'Pretraživanje knjiga po autoru';

        require_once __DIR__ . '/../view/books_search.php';
    }


    public function searchResults()
    {
        if( isset( $_POST['author'] ) && preg_match( '/^[A-Za-z .,]+$/', $_POST['author'] ) )
        {
            $bookList = Book::where( 'author', $_POST['author'] );
            $title = 'Popis knjiga autora ' . $_POST['author'];

            require_once __DIR__ . '/../view/books_index.php';
        }
        else
        {
            $title = 'Došlo je pogreške u unosu imena autora.';
            require_once __DIR__ . '/../view/books_error.php';
        }
    }
}

?>
