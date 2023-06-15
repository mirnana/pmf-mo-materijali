<?php 

require_once __DIR__ . '/../model/loan.class.php';

class LoansController
{
    public function index()
    {
        $title = 'Popis svih posudbi';
        $loanList = Loan::all();

        require_once __DIR__ . '/../view/loans_index.php';
    }


    public function byUser()
    {
        $id_user = $_GET['id'];
        $user = User::find( $id_user );

        $title = 'Popis svih posudbi korisnika ' . $user->name . ' ' . $user->surname;
        $loanList = $user->loans();

        require_once __DIR__ . '/../view/loans_index.php';
    }
}

?>
