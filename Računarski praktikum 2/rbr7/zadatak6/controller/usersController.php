<?php 

class UsersController extends BaseController
{
	public function index() 
	{
		// Kontroler koji prikazuje popis svih usera

		$ls = new LibraryService();

		// Popuni template potrebnim podacima
		$this->registry->template->title = 'Popis svih korisnika knjižnice';
		$this->registry->template->userList = $ls->getAllUsers();

        $this->registry->template->show( 'users_index' );
	}


	public function showLoans()
	{
		// Kontroler koji prikazuje popis svih posuđenih knjiga odabranog usera
		// te omogućuje razduživanje i posuđivanje novih knjiga

		$ls = new LibraryService();

		// Da li nam se šalje (novi) user_id preko post-a? Ako ne, provjeri $_SESSION.
		if( isset( $_POST["user_id"] ) )
		{
			// user_id iz post-a izgleda ovako "user_123" -> pravi id je zapravo samo 123 -> preskoči prvih 5 znakova
			$user_id = substr( $_POST["user_id"], 5 );
		}
		else if( isset( $_SESSION["user_id"] ) )
			$user_id = $_SESSION["user_id"];
		else
		{
			// Nema treće opcije -- nešto ne valja. Preusmjeri na početnu stranicu.
			header( 'Location: ' . __SITE_URL . '/index.php?rt=users' );
			exit;
		}

		// Dohvati podatke o korisniku
		$user = $ls->getUserById( $user_id );
		if( $user === null )
			exit( 'Nema korisnika s id-om ' . $user_id );

		// Stavi ga u $_SESSION tako da uvijek prikazujemo njegove podatke
		$_SESSION[ 'user_id' ] = $user_id;

		// Dohvati sve njegove posudbe
		$loansList = $ls->getLoansByUserId( $user_id );

		// Napravi popis knjiga koje je on posudio.
		// Svaki element book liste je par (knjiga, datum isteka)
		$loanedBookList = array();
		foreach( $loansList as $loan )
			$loanedBookList[] = array( "book" => $ls->getBookById( $loan->id_book ), 
				                       "lease_end" => $loan->lease_end,
				                       "loan_id" => $loan->id );


		// Napravi popis svih knjiga koje su dostupne za posudbu.
		$availableBookList = $ls->getAllAvailableBooks();

		$this->registry->template->user_id = $user_id;
		$this->registry->template->loanedBookList = $loanedBookList;
		$this->registry->template->availableBookList = $availableBookList;
		$this->registry->template->title = 'Popis knjiga korisnika ' . $user->name . ' ' . $user->surname;
        $this->registry->template->show( 'users_showLoans' );
	}	


	public function loanBook()
	{
		// Kontoler koji omogućuje posuđivanje nove knjige tako da obradi podatke iz forme users_showLoans
		// Na kraju samo napravi redirect na početnu stranicu.

		$ls = new LibraryService();

		// Ako nemamo user_id, preusmjeri na početnu stranicu
		if( !isset( $_SESSION['user_id' ] ) )
		{
			header( 'Location: ' . __SITE_URL . '/index.php?rt=users');
			exit();
		}

		$user_id = $_SESSION[ 'user_id' ];

		// Ako nam se ne šalje book_id ispravnog oblika, nešto ne valja
		if( !isset( $_POST['book_id'] ) || !preg_match( '/^book_[0-9]+$/', $_POST['book_id'] ) )
		{
			header( 'Location: ' . __SITE_URL . '/index.php?rt=users');
			exit();
		}

		// Napravi novu posudbu s rokom na mjesec dana
		$book_id = substr( $_POST["book_id"], 5 );
		$ls->makeNewLoan( $user_id, $book_id, date("Y-m-d", time() + 30 * (24 * 60 * 60) ) );

		header( 'Location: ' . __SITE_URL . '/index.php?rt=users/showLoans' );
		exit();
	}


	public function returnBook()
	{
		// Kontoler koji omogućuje vraćanje posuđene knjige tako da obradi podatke iz forme users_showLoans
		// Na kraju samo napravi redirect na početnu stranicu.

		$ls = new LibraryService();

		// Ako nam se ne šalje loan_id ispravnog oblika, nešto ne valja
		if( !isset( $_POST['loan_id'] ) || !preg_match( '/^loan_[0-9]+$/', $_POST['loan_id'] ) )
		{
			header( 'Location: ' . __SITE_URL . '/index.php?rt=users');
			exit();
		}

		// Vrati posuđenu knjigu
		$loan_id = substr( $_POST["loan_id"], 5 );
		$ls->returnLoan( $loan_id );

		header( 'Location: ' . __SITE_URL . '/index.php?rt=users/showLoans' );
		exit();
	}
}; 

?>
