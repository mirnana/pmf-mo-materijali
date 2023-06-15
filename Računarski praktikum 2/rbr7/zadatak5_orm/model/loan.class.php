<?php 

require_once __DIR__ . '/model.class.php';

class Loan extends Model
{
    static protected $table = 'loans';

    public function user()
    {
        return $this->belongsTo( 'User', 'id_user' );
    }

    public function book()
    {
        return $this->belongsTo( 'Book', 'id_book' );
    }
}

?>
