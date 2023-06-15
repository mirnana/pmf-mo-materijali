<?php 

require_once __DIR__ . '/model.class.php';

class User extends Model
{
    static protected $table = 'users';

    public function loans()
    {
        return $this->hasMany( 'Loan', 'id_user' );
    }
}

?>
