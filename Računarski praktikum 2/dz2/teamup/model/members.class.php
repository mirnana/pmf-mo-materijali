<?php 
    class Members {
        
        protected $id
                , $id_project
                , $id_user
                , $member_type;
        
        function __construct ($id, $id_project, $id_user, $member_type) {
            $this->id          = $id;
            $this->id_project  = $id_project;
            $this->id_user     = $id_user;
            $this->member_type = $member_type;
        }

        function __get( $prop ) { 
            return $this->$prop; 
        }
	    function __set( $prop, $val ) { 
            $this->$prop = $val; return $this; 
        }
    }
?>