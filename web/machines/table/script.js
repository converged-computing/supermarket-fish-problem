$(function () {
    $.ajax({
        url: "./data.json",
        type: "GET",
        success: function (data) {
            $("#datatable-json").DataTable({                               
                "ordering": true,
                "paging": true,
                "info": true,
                "scrollX": true,
                "scrollCollapse": true,
                "destroy": true,
                "lengthMenu": [
				    [10, 20, 30, 50, 80, -1],
				    [10, 20, 30, 50, 80, "End"]
			    ],
                data: data,
                columns: [
                    { 
                        data: "environment", 
                    },
                    { 
                        data: "collection", 
                    },
                    { 
                        data: "attribute",
                    },
                    { 
                        data: "value",
                    },
                    { 
                        data: "count",
                    }
                ],
                responsive: true
            });                                 
        }
    });
});
