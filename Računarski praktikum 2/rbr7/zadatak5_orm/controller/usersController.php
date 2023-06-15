<?php 

require_once __DIR__ . '/../model/user.class.php';

class UsersController
{
    public function index()
    {
        $title = 'Popis svih korisnika';
        $userList = User::all();

        require_once __DIR__ . '/../view/users_index.php';
    }
}

?>
